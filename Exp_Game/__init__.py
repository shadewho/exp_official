# Exploratory/Exp_Game/__init__.py

# ##### BEGIN GPL LICENSE BLOCK #####
#

#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# ============================================================================
# CRITICAL: BPY IMPORT GUARD FOR MULTIPROCESSING
# ============================================================================
import sys
_IS_WORKER_PROCESS = (__name__ != '__main__' and 'multiprocessing' in sys.modules)

if not _IS_WORKER_PROCESS:
    import bpy
    from .modal.exp_modal import ExpModal
    from .exp_ui import (
        ExploratoryPanel,
        ExploratoryCharacterPanel,
        ExploratoryProxyMeshPanel,
        VIEW3D_PT_Exploratory_UploadHelper,
        VIEW3D_PT_Exploratory_Performance,
        VIEW3D_PT_Exploratory_PhysicsTuning,
        EXP_OT_FilterCreatePanels,
    )

    # Developer module
    from .developer import (
        register_properties as register_dev_properties,
        unregister_properties as unregister_dev_properties,
        DEV_PT_DeveloperTools,
    )
    from . import developer as dev_module


    from .startup_and_reset.exp_spawn import EXPLORATORY_OT_RemoveCharacter
    from .props_and_utils.exp_properties import (
        remove_scene_properties,
        add_scene_properties,
        CharacterActionsPG,
        ProxyMeshEntry,
        EXPLORATORY_OT_AddProxyMesh,
        EXPLORATORY_OT_RemoveProxyMesh,
        EXPLORATORY_UL_ProxyMeshList,
        CharacterPhysicsConfigPG
    )

    from .reactions.exp_custom_ui import EXPLORE_OT_PreviewCustomText
    from .props_and_utils.exp_utilities import EXPLORATORY_OT_SetGameWorld
    from .props_and_utils import exp_demo_scene
    from .startup_and_reset.exp_startup import EXP_GAME_OT_StartGame
    from .audio.exp_audio import (AUDIO_OT_TestSoundPointer, CharacterAudioPG, EXPLORATORY_OT_BuildAudio,
                            EXP_AUDIO_OT_LoadAudioFile, EXP_AUDIO_OT_TestReactionSound, EXP_AUDIO_OT_PackAllSounds,
                            EXP_AUDIO_OT_LoadCharacterAudioFile
    )

    from .interactions.exp_interaction_definition import (InteractionDefinition, register_interaction_properties,
                                                           unregister_interaction_properties, ReactionLinkPG,)

    from .interactions.exp_interactions import (
        EXPLORATORY_OT_AddInteraction,
        EXPLORATORY_OT_RemoveInteraction,
        EXPLORATORY_OT_AddReactionToInteraction,
        EXPLORATORY_OT_RemoveReactionFromInteraction,
        EXPLORATORY_OT_CreateReactionAndLink,
        EXPLORATORY_OT_DuplicateInteraction,
    )

    from .reactions.exp_reaction_definition import ReactionDefinition, unregister_reaction_library_properties, EXPLORATORY_OT_DuplicateGlobalReaction
    from .systems.exp_counters_timers import (
        register_counter_timer_properties,
        unregister_counter_timer_properties
    )

    from .systems.exp_performance import (
        register   as register_performance,
        unregister as unregister_performance
    )
    from .reactions.exp_mobility_and_game_reactions import (
        MobilityReactionsPG,
        MeshVisibilityReactionsPG,
    )
    from .startup_and_reset.exp_game_reset import EXPLORATORY_OT_ResetGame
    from .props_and_utils.exp_upload_helper import register as register_upload_helper, unregister as unregister_upload_helper

    from .reactions.exp_action_keys import ActionKeyItemPG, register_action_key_properties, unregister_action_key_properties

    # --- Engine testing ---
    from .engine.testing.test_operator import EXP_ENGINE_OT_StressTest, EXP_ENGINE_OT_QuickTest

    # --- Animation Engine ---
    from .engine import animations as anim_module

    # --- Animation 2.0 Operators & Properties ---
    from .animations import test_panel as anim2_test_panel
    from .animations import rig_probe as rig_probe_module

    # --- Pose Library ---
    from .animations import pose_library as pose_lib

    # ============================================================================
    # START GAME KEYMAP MANAGEMENT
    # ============================================================================
    _addon_keymaps = []

    def register_start_game_keymap():
        """Register the Start Game keymap for VIEW_3D."""
        wm = bpy.context.window_manager
        kc = wm.keyconfigs.addon
        if kc is None:
            return

        # Get the preferred key from addon preferences
        try:
            prefs = bpy.context.preferences.addons["Exploratory"].preferences
            key = getattr(prefs, "key_start_game", "P")
        except Exception:
            key = "P"

        # Create keymap for 3D View
        km = kc.keymaps.new(name='3D View', space_type='VIEW_3D')
        kmi = km.keymap_items.new(
            "exploratory.start_game",
            type=key,
            value='PRESS'
        )
        _addon_keymaps.append((km, kmi))

    def unregister_start_game_keymap():
        """Unregister all addon keymaps."""
        for km, kmi in _addon_keymaps:
            try:
                km.keymap_items.remove(kmi)
            except Exception:
                pass
        _addon_keymaps.clear()

    def update_start_game_keymap():
        """
        Called when the user changes the keybind in preferences.
        Removes old keymap and registers new one with updated key.
        """
        unregister_start_game_keymap()
        register_start_game_keymap()

    def register():
        # --- Developer Properties (early registration) ---
        register_dev_properties()

        # --- Developer Module (rig analyzer, etc.) ---
        dev_module.register()

        # --- Mobility / Mesh Visibility PGs (must be registered before ReactionDefinition) ---
        bpy.utils.register_class(MobilityReactionsPG)
        bpy.utils.register_class(MeshVisibilityReactionsPG)
        bpy.types.Scene.mobility_game = bpy.props.PointerProperty(type=MobilityReactionsPG)
        # --- Reset Game ---
        bpy.utils.register_class(EXPLORATORY_OT_ResetGame)

        # --- action keys ---
        bpy.utils.register_class(ActionKeyItemPG)
        register_action_key_properties()

        # --- Interactions & Reactions ---
        bpy.utils.register_class(ReactionDefinition)   # must be before scene.reactions is attached
        bpy.utils.register_class(ReactionLinkPG)       # link PG used inside InteractionDefinition
        bpy.utils.register_class(InteractionDefinition)
        bpy.utils.register_class(EXPLORATORY_OT_AddInteraction)
        bpy.utils.register_class(EXPLORATORY_OT_RemoveInteraction)
        bpy.utils.register_class(EXPLORATORY_OT_AddReactionToInteraction)
        bpy.utils.register_class(EXPLORATORY_OT_RemoveReactionFromInteraction)
        bpy.utils.register_class(EXPLORATORY_OT_CreateReactionAndLink)
        register_interaction_properties()


        # --- Panels & UILists (Ordered) ---
        bpy.utils.register_class(ExploratoryPanel)

        bpy.utils.register_class(ExploratoryCharacterPanel)

        bpy.utils.register_class(ExploratoryProxyMeshPanel)

        bpy.utils.register_class(VIEW3D_PT_Exploratory_UploadHelper)

        bpy.utils.register_class(VIEW3D_PT_Exploratory_Performance)

        bpy.utils.register_class(DEV_PT_DeveloperTools)

        bpy.utils.register_class(VIEW3D_PT_Exploratory_PhysicsTuning)

        bpy.utils.register_class(EXP_OT_FilterCreatePanels)


        bpy.utils.register_class(EXPLORE_OT_PreviewCustomText)

        # Register the UILists used in panels
        bpy.utils.register_class(EXPLORATORY_OT_DuplicateInteraction)
        bpy.utils.register_class(EXPLORATORY_OT_DuplicateGlobalReaction)

        # --- Modal & Game Start Operators ---
        bpy.utils.register_class(ExpModal)
        bpy.utils.register_class(EXP_GAME_OT_StartGame)

        # --- Audio Operators & Properties ---
        bpy.utils.register_class(AUDIO_OT_TestSoundPointer)
        bpy.utils.register_class(EXP_AUDIO_OT_LoadCharacterAudioFile)
        bpy.utils.register_class(CharacterAudioPG)
        bpy.utils.register_class(EXPLORATORY_OT_BuildAudio)
        bpy.utils.register_class(EXP_AUDIO_OT_LoadAudioFile)
        bpy.utils.register_class(EXP_AUDIO_OT_TestReactionSound)
        bpy.utils.register_class(EXP_AUDIO_OT_PackAllSounds)
        bpy.types.Scene.character_audio = bpy.props.PointerProperty(type=CharacterAudioPG)

        # --- Character Removal ---
        bpy.utils.register_class(EXPLORATORY_OT_RemoveCharacter)

        # --- Counter & Timer Properties & Operators ---
        register_counter_timer_properties()

        # --- Character Actions & Proxy Mesh Properties ---
        bpy.utils.register_class(CharacterActionsPG)
        bpy.utils.register_class(ProxyMeshEntry)
        bpy.utils.register_class(EXPLORATORY_UL_ProxyMeshList)
        bpy.utils.register_class(CharacterPhysicsConfigPG)
        bpy.utils.register_class(EXPLORATORY_OT_AddProxyMesh)
        bpy.utils.register_class(EXPLORATORY_OT_RemoveProxyMesh)
        bpy.types.Scene.character_actions = bpy.props.PointerProperty(type=CharacterActionsPG)

        # --- Scene Properties ---
        add_scene_properties()


        bpy.utils.register_class(EXPLORATORY_OT_SetGameWorld)

        # ─── Demo scene ──────────────────────────────────
        exp_demo_scene.register()

        # ─── Performance Culling ──────────────────────────────────
        register_performance()

        register_upload_helper()

        # ─── Engine Testing Operators ──────────────────────────────
        bpy.utils.register_class(EXP_ENGINE_OT_QuickTest)
        bpy.utils.register_class(EXP_ENGINE_OT_StressTest)

        # ─── Animation Engine ──────────────────────────────
        anim_module.register()

        # ─── Animation 2.0 Operators & Properties ──────────────────────────────
        anim2_test_panel.register()
        rig_probe_module.register()

        # ─── Pose Library ──────────────────────────────
        pose_lib.register_pose_library()

        # ─── Start Game Keymap ──────────────────────────────
        register_start_game_keymap()


    def unregister():
        # ─── Start Game Keymap (unregister first) ──────────────────────────────
        unregister_start_game_keymap()
        remove_scene_properties()
        unregister_interaction_properties()
        unregister_counter_timer_properties()
        unregister_reaction_library_properties()
        unregister_action_key_properties()
        # --- Mobility / Mesh Visibility PGs ---
        if hasattr(bpy.types.Scene, 'mobility_game'):
            del bpy.types.Scene.mobility_game
        bpy.utils.unregister_class(MeshVisibilityReactionsPG)
        bpy.utils.unregister_class(MobilityReactionsPG)

        bpy.utils.unregister_class(EXPLORATORY_OT_ResetGame)
        bpy.utils.unregister_class(EXPLORATORY_OT_RemoveCharacter)

        # --- Audio ---
        if hasattr(bpy.types.Scene, 'character_audio'):
            del bpy.types.Scene.character_audio
        bpy.utils.unregister_class(EXP_AUDIO_OT_PackAllSounds)
        bpy.utils.unregister_class(EXP_AUDIO_OT_TestReactionSound)
        bpy.utils.unregister_class(EXP_AUDIO_OT_LoadAudioFile)
        bpy.utils.unregister_class(AUDIO_OT_TestSoundPointer)
        bpy.utils.unregister_class(EXP_AUDIO_OT_LoadCharacterAudioFile)
        bpy.utils.unregister_class(CharacterAudioPG)
        bpy.utils.unregister_class(EXPLORATORY_OT_BuildAudio)


        # --- Modal & Game Start ---
        bpy.utils.unregister_class(ExpModal)
        bpy.utils.unregister_class(EXP_GAME_OT_StartGame)

        # --- Panels & UILists ---
        bpy.utils.unregister_class(EXPLORATORY_OT_DuplicateGlobalReaction)
        bpy.utils.unregister_class(ExploratoryPanel)
        bpy.utils.unregister_class(VIEW3D_PT_Exploratory_UploadHelper)
        bpy.utils.unregister_class(VIEW3D_PT_Exploratory_Performance)
        bpy.utils.unregister_class(DEV_PT_DeveloperTools)
        bpy.utils.unregister_class(ExploratoryCharacterPanel)
        bpy.utils.unregister_class(ExploratoryProxyMeshPanel)
        bpy.utils.unregister_class(EXPLORE_OT_PreviewCustomText)
        bpy.utils.unregister_class(EXP_OT_FilterCreatePanels)

        # --- actions ---
        bpy.utils.unregister_class(ActionKeyItemPG)

        # --- Interactions & Reactions ---
        bpy.utils.unregister_class(EXPLORATORY_OT_RemoveReactionFromInteraction)
        bpy.utils.unregister_class(EXPLORATORY_OT_AddReactionToInteraction)
        bpy.utils.unregister_class(EXPLORATORY_OT_RemoveInteraction)
        bpy.utils.unregister_class(EXPLORATORY_OT_AddInteraction)
        bpy.utils.unregister_class(InteractionDefinition)
        bpy.utils.unregister_class(ReactionLinkPG)
        bpy.utils.unregister_class(ReactionDefinition)
        bpy.utils.unregister_class(EXPLORATORY_OT_CreateReactionAndLink)


        bpy.utils.unregister_class(EXPLORATORY_OT_DuplicateInteraction)

        bpy.utils.unregister_class(EXPLORATORY_OT_SetGameWorld)


        # --- Character Actions & Proxy Mesh ---
        if hasattr(bpy.types.Scene, 'character_actions'):
            del bpy.types.Scene.character_actions
        bpy.utils.unregister_class(EXPLORATORY_OT_RemoveProxyMesh)
        bpy.utils.unregister_class(CharacterActionsPG)
        bpy.utils.unregister_class(CharacterPhysicsConfigPG)
        bpy.utils.unregister_class(ProxyMeshEntry)
        bpy.utils.unregister_class(EXPLORATORY_UL_ProxyMeshList)
        bpy.utils.unregister_class(EXPLORATORY_OT_AddProxyMesh)
        bpy.utils.unregister_class(VIEW3D_PT_Exploratory_PhysicsTuning)


        # ─── DEMO DEMO ──────────────────────────────────
        exp_demo_scene.unregister()

        # ─── Engine Testing Operators ──────────────────────────────
        bpy.utils.unregister_class(EXP_ENGINE_OT_StressTest)
        bpy.utils.unregister_class(EXP_ENGINE_OT_QuickTest)

        # ─── Performance Culling ──────────────────────────────────
        unregister_performance()

        # ─── Pose Library ──────────────────────────────
        pose_lib.unregister_pose_library()

        # ─── Animation 2.0 Operators & Properties ──────────────────────────────
        rig_probe_module.unregister()
        anim2_test_panel.unregister()

        # ─── Animation Engine ──────────────────────────────
        anim_module.unregister()

        # ─── Developer Module ──────────────────────────────────
        dev_module.unregister()

        # ─── Developer Properties ──────────────────────────────────
        unregister_dev_properties()

        unregister_upload_helper()

else:
    # Worker process - provide stub functions
    def register():
        pass

    def unregister():
        pass

if __name__ == "__main__":
    register()