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


import bpy
from .modal.exp_modal import ExpModal
from .exp_ui import (
    ExploratoryPanel,
    ExploratoryCharacterPanel,
    ExploratoryProxyMeshPanel,
    VIEW3D_PT_Exploratory_Studio,
    EXPLORATORY_UL_CustomInteractions,
    EXPLORATORY_UL_ReactionsInInteraction,
    VIEW3D_PT_Objectives,
    EXPLORATORY_UL_Objectives,
    VIEW3D_PT_Exploratory_UploadHelper,
    VIEW3D_PT_Exploratory_Performance,
    VIEW3D_PT_Exploratory_PhysicsTuning,
    EXP_OT_FilterCreatePanels,
    EXPLORATORY_UL_ReactionLibrary,
    EXPLORATORY_OT_AddGlobalReaction,
    EXPLORATORY_OT_RemoveGlobalReaction,
    VIEW3D_PT_Exploratory_Reactions,
    EXPLORATORY_OT_DuplicateGlobalReaction,
)


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
from .reactions.exp_reaction_definition import ReactionDefinition
from .systems.exp_objectives import (
    ObjectiveDefinition,
    EXPLORATORY_OT_AddObjective,
    EXPLORATORY_OT_RemoveObjective,
    register_objective_properties,
    unregister_objective_properties
)

from .systems.exp_performance import (
    register   as register_performance,
    unregister as unregister_performance
)
from .reactions.exp_mobility_and_game_reactions import MobilityGameReactionsPG
from .startup_and_reset.exp_game_reset import EXPLORATORY_OT_ResetGame
from .props_and_utils.exp_upload_helper import register as register_upload_helper, unregister as unregister_upload_helper


def register():
    # --- Mobility Game Reactions ---
    bpy.utils.register_class(MobilityGameReactionsPG)
    bpy.types.Scene.mobility_game = bpy.props.PointerProperty(type=MobilityGameReactionsPG)

    # --- Reset Game ---
    bpy.utils.register_class(EXPLORATORY_OT_ResetGame)

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

    bpy.utils.register_class(VIEW3D_PT_Exploratory_Studio)

    bpy.utils.register_class(VIEW3D_PT_Objectives) 

    bpy.utils.register_class(VIEW3D_PT_Exploratory_UploadHelper)

    bpy.utils.register_class(VIEW3D_PT_Exploratory_Performance)

    bpy.utils.register_class(VIEW3D_PT_Exploratory_PhysicsTuning)

    bpy.utils.register_class(EXP_OT_FilterCreatePanels)


    bpy.utils.register_class(EXPLORE_OT_PreviewCustomText)

    # Register the UILists used in panels
    bpy.utils.register_class(EXPLORATORY_UL_CustomInteractions)
    bpy.utils.register_class(EXPLORATORY_OT_DuplicateInteraction)
    bpy.utils.register_class(EXPLORATORY_UL_ReactionsInInteraction)
    bpy.utils.register_class(EXPLORATORY_OT_DuplicateGlobalReaction)
    bpy.utils.register_class(EXPLORATORY_UL_Objectives)

    bpy.utils.register_class(EXPLORATORY_UL_ReactionLibrary)
    bpy.utils.register_class(EXPLORATORY_OT_AddGlobalReaction)
    bpy.utils.register_class(EXPLORATORY_OT_RemoveGlobalReaction)
    bpy.utils.register_class(VIEW3D_PT_Exploratory_Reactions)

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

    # --- Objectives Properties & Operators ---
    register_objective_properties()

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

def unregister():
    remove_scene_properties()
    unregister_interaction_properties()
    unregister_objective_properties()

    # --- Mobility Game Reactions ---
    del bpy.types.Scene.mobility_game
    bpy.utils.unregister_class(MobilityGameReactionsPG)
    bpy.utils.unregister_class(EXPLORATORY_OT_ResetGame)
    bpy.utils.unregister_class(EXPLORATORY_OT_RemoveCharacter)

    # --- Audio ---
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
    bpy.utils.unregister_class(VIEW3D_PT_Exploratory_Reactions)
    bpy.utils.unregister_class(EXPLORATORY_OT_RemoveGlobalReaction)
    bpy.utils.unregister_class(EXPLORATORY_OT_AddGlobalReaction)
    bpy.utils.unregister_class(EXPLORATORY_UL_ReactionLibrary)
    bpy.utils.unregister_class(EXPLORATORY_UL_ReactionsInInteraction)
    bpy.utils.unregister_class(EXPLORATORY_UL_CustomInteractions)
    bpy.utils.unregister_class(VIEW3D_PT_Exploratory_Studio)
    bpy.utils.unregister_class(ExploratoryPanel)
    bpy.utils.unregister_class(VIEW3D_PT_Objectives)
    bpy.utils.unregister_class(VIEW3D_PT_Exploratory_UploadHelper)
    bpy.utils.unregister_class(VIEW3D_PT_Exploratory_Performance)
    bpy.utils.unregister_class(EXPLORATORY_UL_Objectives)
    bpy.utils.unregister_class(ExploratoryCharacterPanel)
    bpy.utils.unregister_class(ExploratoryProxyMeshPanel)
    bpy.utils.unregister_class(EXPLORE_OT_PreviewCustomText)
    bpy.utils.unregister_class(EXP_OT_FilterCreatePanels)


    # --- Interactions & Reactions ---
    bpy.utils.unregister_class(EXPLORATORY_OT_RemoveReactionFromInteraction)
    bpy.utils.unregister_class(EXPLORATORY_OT_AddReactionToInteraction)
    bpy.utils.unregister_class(EXPLORATORY_OT_RemoveInteraction)
    bpy.utils.unregister_class(EXPLORATORY_OT_AddInteraction)
    bpy.utils.unregister_class(InteractionDefinition)
    bpy.utils.unregister_class(ReactionLinkPG)
    bpy.utils.unregister_class(ReactionDefinition)
    bpy.utils.unregister_class(EXPLORATORY_OT_DuplicateInteraction)
    bpy.utils.unregister_class(EXPLORATORY_OT_DuplicateInteraction)

    bpy.utils.unregister_class(EXPLORATORY_OT_SetGameWorld)


    # --- Character Actions & Proxy Mesh ---
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

     # ─── Performance Culling ──────────────────────────────────
    unregister_performance()

    unregister_upload_helper()

if __name__ == "__main__":
    register()
