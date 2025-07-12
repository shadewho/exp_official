# Exploratory/Exp_Game/__init__.py
import bpy
from .exp_modal import ExpModal
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
    VIEW3D_PT_Exploratory_Performance
)


from .exp_spawn import EXPLORATORY_OT_RemoveCharacter
from .exp_properties import (
    remove_scene_properties,
    add_scene_properties,
    CharacterActionsPG,
    ProxyMeshEntry,
    EXPLORATORY_OT_AddProxyMesh,
    EXPLORATORY_OT_RemoveProxyMesh,
    EXPLORATORY_UL_ProxyMeshList,
)
from .exp_custom_ui import EXPLORE_OT_PreviewCustomText
from .exp_utilities import EXPLORATORY_OT_SetGameWorld
from .exp_startup import EXP_GAME_OT_StartGame
from .exp_audio import (AUDIO_OT_TestSoundPointer, CharacterAudioPG, EXPLORATORY_OT_BuildAudio,
                        EXP_AUDIO_OT_LoadAudioFile, EXP_AUDIO_OT_TestReactionSound, EXP_AUDIO_OT_PackAllSounds,
                        EXP_AUDIO_OT_LoadCharacterAudioFile
)

from .exp_interaction_definition import InteractionDefinition, register_interaction_properties, unregister_interaction_properties

from .exp_interactions import (
    EXPLORATORY_OT_AddInteraction,
    EXPLORATORY_OT_RemoveInteraction,
    EXPLORATORY_OT_AddReactionToInteraction,
    EXPLORATORY_OT_RemoveReactionFromInteraction,
)
from .exp_reaction_definition import ReactionDefinition
from .exp_objectives import (
    ObjectiveDefinition,
    EXPLORATORY_OT_AddObjective,
    EXPLORATORY_OT_RemoveObjective,
    register_objective_properties,
    unregister_objective_properties
)

from .exp_performance import (
    register   as register_performance,
    unregister as unregister_performance
)
from .exp_mobility_and_game_reactions import MobilityGameReactionsPG
from .exp_game_reset import EXPLORATORY_OT_ResetGame
from .exp_upload_helper import register as register_upload_helper, unregister as unregister_upload_helper


def register():
    # --- Mobility Game Reactions ---
    bpy.utils.register_class(MobilityGameReactionsPG)
    bpy.types.Scene.mobility_game = bpy.props.PointerProperty(type=MobilityGameReactionsPG)

    # --- Reset Game ---
    bpy.utils.register_class(EXPLORATORY_OT_ResetGame)

    # --- Interactions & Reactions ---
    bpy.utils.register_class(ReactionDefinition)
    bpy.utils.register_class(InteractionDefinition)
    bpy.utils.register_class(EXPLORATORY_OT_AddInteraction)
    bpy.utils.register_class(EXPLORATORY_OT_RemoveInteraction)
    bpy.utils.register_class(EXPLORATORY_OT_AddReactionToInteraction)
    bpy.utils.register_class(EXPLORATORY_OT_RemoveReactionFromInteraction)
    register_interaction_properties()

    # --- Panels & UILists (Ordered) ---
    # 1. Main Panel (mode toggle)
    bpy.utils.register_class(ExploratoryPanel)
    # 2. Character, Actions, Audio Panel (CREATE mode)
    bpy.utils.register_class(ExploratoryCharacterPanel)
    # 3. Proxy Meshes Panel (CREATE mode)
    bpy.utils.register_class(ExploratoryProxyMeshPanel)
    # 4. Custom Interactions Panel (CREATE mode)
    bpy.utils.register_class(VIEW3D_PT_Exploratory_Studio)
    # 5. Objectives Panel (CREATE mode)
    bpy.utils.register_class(VIEW3D_PT_Objectives) 
    #6 Uploads
    bpy.utils.register_class(VIEW3D_PT_Exploratory_UploadHelper)

    bpy.utils.register_class(VIEW3D_PT_Exploratory_Performance)

    bpy.utils.register_class(EXPLORE_OT_PreviewCustomText)

    # Register the UILists used in panels
    bpy.utils.register_class(EXPLORATORY_UL_CustomInteractions)
    bpy.utils.register_class(EXPLORATORY_UL_ReactionsInInteraction)
    bpy.utils.register_class(EXPLORATORY_UL_Objectives)

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
    bpy.utils.register_class(EXPLORATORY_OT_AddProxyMesh)
    bpy.utils.register_class(EXPLORATORY_OT_RemoveProxyMesh)
    bpy.types.Scene.character_actions = bpy.props.PointerProperty(type=CharacterActionsPG)
    
    # --- Scene Properties ---
    add_scene_properties()

    bpy.utils.register_class(EXPLORATORY_OT_SetGameWorld)

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

    # --- Interactions & Reactions ---
    bpy.utils.unregister_class(EXPLORATORY_OT_RemoveReactionFromInteraction)
    bpy.utils.unregister_class(EXPLORATORY_OT_AddReactionToInteraction)
    bpy.utils.unregister_class(EXPLORATORY_OT_RemoveInteraction)
    bpy.utils.unregister_class(EXPLORATORY_OT_AddInteraction)
    bpy.utils.unregister_class(InteractionDefinition)
    bpy.utils.unregister_class(ReactionDefinition)

    bpy.utils.unregister_class(EXPLORATORY_OT_SetGameWorld)

    # --- Character Actions & Proxy Mesh ---
    del bpy.types.Scene.character_actions
    bpy.utils.unregister_class(EXPLORATORY_OT_RemoveProxyMesh)
    bpy.utils.unregister_class(CharacterActionsPG)
    bpy.utils.unregister_class(ProxyMeshEntry)
    bpy.utils.unregister_class(EXPLORATORY_UL_ProxyMeshList)
    bpy.utils.unregister_class(EXPLORATORY_OT_AddProxyMesh)

     # ─── Performance Culling ──────────────────────────────────
    unregister_performance()

    unregister_upload_helper()

if __name__ == "__main__":
    register()
