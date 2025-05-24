import os

class SAM2SymlinkStreamer:
    def __init__(self, predictor, symlink_dir):
        """Initialize a streaming processor using a symlink directory"""
        self.predictor = predictor
        self.symlink_dir = symlink_dir
        os.makedirs(symlink_dir, exist_ok=True)
        self.inference_state = None
        self.frame_mapping = {}  # Maps current symlink indices to source frame paths
        # self.current_contents = set() # Can be derived from self.frame_mapping.keys()
        
    def update_symlinks(self, frame_dict, new_prompts=None):
        """
        Update symlinks and process video
        
        Args:
            frame_dict: Dictionary mapping desired frame indices (0-indexed for current window) to source file paths
                        {0: '/path/to/frameX.jpg', 1: '/path/to/frameY.jpg', ...}
            new_prompts: Optional dictionary with new prompts for specific frames (using current window's 0-indexed frame_idx)
                        {frame_idx: {obj_id: {'points': points, 'labels': labels}}}
        
        Returns:
            Dictionary with segmentation results
        """
        # Clear existing symlinks that are not in the new set of source paths
        # This needs to be more robust if symlink names don't match frame_dict keys directly
        # For now, assume symlink names are "xxxxx.jpg" matching frame_dict keys
        current_symlink_filenames = {f"{idx:05d}.jpg" for idx in frame_dict.keys()}
        for f_name in os.listdir(self.symlink_dir):
            if f_name.endswith(('.jpg', '.jpeg', '.png')):
                # A more robust cleanup would be to remove all and recreate,
                # or check if the target of the symlink is still needed.
                # Simple approach: remove symlinks whose names (derived from new indices) are not in frame_dict
                # This part of the original code was:
                # if f.endswith(('.jpg', '.jpeg', '.png')):
                #    idx = int(os.path.splitext(f)[0])
                #    if idx not in new_indices: # new_indices were frame_dict.keys()
                #        os.remove(os.path.join(self.symlink_dir, f))
                # This seems fine if symlinks are always named based on frame_dict keys.
                # Let's ensure symlinks are named based on frame_dict keys.
                pass # Symlink creation below will overwrite or create new ones. Old ones not in frame_dict can be removed.

        # Create/Update symlinks
        # First, remove all old symlinks to prevent conflicts if names change meaning
        for f_name in os.listdir(self.symlink_dir):
            if os.path.islink(os.path.join(self.symlink_dir, f_name)):
                os.remove(os.path.join(self.symlink_dir, f_name))

        for idx, src_path in frame_dict.items():
            symlink_path = os.path.join(self.symlink_dir, f"{idx:05d}.jpg")
            # if os.path.exists(symlink_path) or os.path.islink(symlink_path): # Check for islink too
            #     os.remove(symlink_path) # Already removed above
            os.symlink(os.path.abspath(src_path), symlink_path)
        
        if self.inference_state is None:
            print("Self.inference_state is None, initializing...")
            self.inference_state = self.predictor.init_state(video_path=self.symlink_dir)
            self.frame_mapping = frame_dict.copy() 
            
            if new_prompts:
                self._apply_prompts(new_prompts)
                
            results = self._run_propagation()
            return results
            
        # For subsequent runs, preserve and transfer tracking information
        old_predictor_fields = {
            "obj_id_to_idx": self.inference_state.get("obj_id_to_idx", {}).copy(),
            "obj_idx_to_id": self.inference_state.get("obj_idx_to_id", {}).copy(),
            "obj_ids": self.inference_state.get("obj_ids", []).copy(),
            "point_inputs_per_obj": {k: v.copy() for k, v in self.inference_state.get("point_inputs_per_obj", {}).items()},
            "mask_inputs_per_obj": {k: v.copy() for k, v in self.inference_state.get("mask_inputs_per_obj", {}).items()},
            "output_dict_per_obj": {
                k: {
                    "cond_frame_outputs": v.get("cond_frame_outputs", {}).copy(),
                    "non_cond_frame_outputs": v.get("non_cond_frame_outputs", {}).copy()
                } for k, v in self.inference_state.get("output_dict_per_obj", {}).items()
            },
            "frames_tracked_per_obj": {k: v.copy() for k, v in self.inference_state.get("frames_tracked_per_obj", {}).items()},
            # temp_output_dict_per_obj is usually transient within predictor calls, might not need deep copy for transfer
        }
        old_frame_mapping = self.frame_mapping.copy() 
        
        # Initialize new state with updated symlinks
        self.inference_state = self.predictor.init_state(video_path=self.symlink_dir)
        self.frame_mapping = frame_dict.copy() # New mapping for current symlinks
        
        # Transfer relevant tracking info
        self._transfer_tracking_info(old_predictor_fields, old_frame_mapping, self.frame_mapping)
        
        # Apply new prompts if any
        if new_prompts:
            self._apply_prompts(new_prompts)
        
        # Run propagation
        results = self._run_propagation()
        
        return results
        
    def _transfer_tracking_info(self, old_fields, old_symlink_idx_to_path, new_symlink_idx_to_path):
        """Transfer tracking info from old state to new state based on matching source paths."""
        
        # Copy basic object mappings
        self.inference_state["obj_id_to_idx"] = old_fields["obj_id_to_idx"]
        self.inference_state["obj_idx_to_id"] = old_fields["obj_idx_to_id"]
        self.inference_state["obj_ids"] = old_fields["obj_ids"]
        
        if not self.inference_state["obj_ids"]:
            return

        num_new_frames = len(self.inference_state["images"])

        # Ensure per-object dictionaries exist in the new state for all known objects
        for obj_idx_str in old_fields["obj_idx_to_id"].keys():
            obj_idx = int(obj_idx_str)
            if obj_idx not in self.inference_state["point_inputs_per_obj"]: self.inference_state["point_inputs_per_obj"][obj_idx] = {}
            if obj_idx not in self.inference_state["mask_inputs_per_obj"]: self.inference_state["mask_inputs_per_obj"][obj_idx] = {}
            if obj_idx not in self.inference_state["output_dict_per_obj"]:
                self.inference_state["output_dict_per_obj"][obj_idx] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
            if obj_idx not in self.inference_state["frames_tracked_per_obj"]: self.inference_state["frames_tracked_per_obj"][obj_idx] = {}
            # temp_output_dict_per_obj is managed by the predictor during its operations.
            # It's usually initialized when _obj_id_to_idx is called (e.g. by add_new_points_or_box).
            # If an object has no new prompts, its temp_output_dict might not be auto-initialized by predictor.
            # Predictor's preflight check iterates obj_idx from 0 to N-1.
            if obj_idx not in self.inference_state.get("temp_output_dict_per_obj", {}):
                 if "temp_output_dict_per_obj" not in self.inference_state: self.inference_state["temp_output_dict_per_obj"] = {}
                 self.inference_state["temp_output_dict_per_obj"][obj_idx] = {"cond_frame_outputs": {},"non_cond_frame_outputs": {}}


        for obj_idx_str, _ in old_fields["obj_idx_to_id"].items():
            obj_idx = int(obj_idx_str)

            # Helper to transfer data for a given field
            def transfer_field_data(field_name, sub_field_name=None):
                new_data_map = {}
                old_obj_specific_data = old_fields.get(field_name, {}).get(obj_idx, {})
                if sub_field_name: # For nested dicts like output_dict_per_obj
                    old_obj_specific_data = old_obj_specific_data.get(sub_field_name, {})

                for old_sym_idx, data_item in old_obj_specific_data.items():
                    old_path = old_symlink_idx_to_path.get(old_sym_idx)
                    if old_path:
                        for new_sym_idx, new_path in new_symlink_idx_to_path.items():
                            if new_path == old_path and new_sym_idx < num_new_frames:
                                new_data_map[new_sym_idx] = data_item
                                break
                
                if sub_field_name:
                    self.inference_state[field_name][obj_idx][sub_field_name] = new_data_map
                else:
                    self.inference_state[field_name][obj_idx] = new_data_map

            transfer_field_data("point_inputs_per_obj")
            transfer_field_data("mask_inputs_per_obj")
            transfer_field_data("output_dict_per_obj", "cond_frame_outputs")
            transfer_field_data("output_dict_per_obj", "non_cond_frame_outputs")
            transfer_field_data("frames_tracked_per_obj")
    
    def _apply_prompts(self, prompts_dict):
        """Apply prompts to specific frames"""
        for frame_idx, frame_prompts in prompts_dict.items():
            for obj_id, prompt in frame_prompts.items():
                # For click prompts
                if "points" in prompt and "labels" in prompt:
                    self.predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=prompt["points"],
                        labels=prompt["labels"]
                    )
                # For box prompts
                elif "box" in prompt:
                    self.predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        box=prompt["box"]
                    )
                # For mask prompts
                elif "mask" in prompt:
                    self.predictor.add_new_mask(
                        inference_state=self.inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        mask=prompt["mask"]
                    )
    
    def _run_propagation(self):
        """Run propagation and collect results"""
        results = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            results[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        return results