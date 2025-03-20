
#init.py

import bpy
from .exp_modal import ExpModal
from .exp_ui import (ExploratoryPanel,
                    VIEW3D_PT_Exploratory_Studio, 
                    EXPLORATORY_UL_CustomInteractions,
                    EXPLORATORY_UL_ReactionsInInteraction,
                    VIEW3D_PT_Objectives,
                    EXPLORATORY_UL_Objectives,
                    ExploratoryCharacterPanel,
                    ExploratoryProxyMeshPanel
)

from .exp_properties import (remove_scene_properties, add_scene_properties,
                             CharacterActionsPG, ProxyMeshEntry, EXPLORATORY_OT_AddProxyMesh,
                             EXPLORATORY_OT_RemoveProxyMesh, EXPLORATORY_UL_ProxyMeshList
)

from .exp_startup import EXP_GAME_OT_StartGame
from .exp_audio import AUDIO_OT_TestSoundPointer, CharacterAudioPG, EXPLORATORY_OT_BuildAudio

from .exp_interactions import (
    InteractionDefinition,
    EXPLORATORY_OT_AddInteraction,
    EXPLORATORY_OT_RemoveInteraction,
    EXPLORATORY_OT_AddReactionToInteraction,
    EXPLORATORY_OT_RemoveReactionFromInteraction,
    register_interaction_properties,
    unregister_interaction_properties,
)

from .exp_reactions import (
    ReactionDefinition
)

from .exp_objectives import (ObjectiveDefinition,
                                EXPLORATORY_OT_AddObjective,
                                EXPLORATORY_OT_RemoveObjective,
                                register_objective_properties,
                                unregister_objective_properties
    )
from .exp_mobility_and_game_reactions import (
    MobilityGameReactionsPG
)
from .exp_game_reset import EXPLORATORY_OT_ResetGame


def register():

    #mobility game reactions
    bpy.utils.register_class(MobilityGameReactionsPG)
    bpy.types.Scene.mobility_game = bpy.props.PointerProperty(
        type=MobilityGameReactionsPG
    )
    #reset game
    bpy.utils.register_class(EXPLORATORY_OT_ResetGame)

    # 1) ReactionDefinition first
    bpy.utils.register_class(ReactionDefinition)

    # 2) Then InteractionDefinition
    bpy.utils.register_class(InteractionDefinition)

    # 3) Now any operators or classes using InteractionDefinition
    bpy.utils.register_class(EXPLORATORY_OT_AddInteraction)
    bpy.utils.register_class(EXPLORATORY_OT_RemoveInteraction)
    bpy.utils.register_class(EXPLORATORY_OT_AddReactionToInteraction)
    bpy.utils.register_class(EXPLORATORY_OT_RemoveReactionFromInteraction)

    # 4) Register the Scene props that reference InteractionDefinition
    register_interaction_properties()

    # 5) Then the rest (panels, UILists, etc.)
    bpy.utils.register_class(ExploratoryPanel)
    bpy.utils.register_class(VIEW3D_PT_Exploratory_Studio)
    bpy.utils.register_class(EXPLORATORY_UL_CustomInteractions)
    bpy.utils.register_class(EXPLORATORY_UL_ReactionsInInteraction)
    bpy.utils.register_class(VIEW3D_PT_Objectives)
    bpy.utils.register_class(EXPLORATORY_UL_Objectives)
    bpy.utils.register_class(ExploratoryCharacterPanel)
    bpy.utils.register_class(ExploratoryProxyMeshPanel)


    bpy.utils.register_class(ExpModal)
    bpy.utils.register_class(EXP_GAME_OT_StartGame)


    bpy.utils.register_class(AUDIO_OT_TestSoundPointer)
    bpy.utils.register_class(CharacterAudioPG)
    bpy.utils.register_class(EXPLORATORY_OT_BuildAudio)
    bpy.types.Scene.character_audio = bpy.props.PointerProperty(type=CharacterAudioPG)

    #objectives
    register_objective_properties()

    bpy.utils.register_class(CharacterActionsPG)
    bpy.utils.register_class(ProxyMeshEntry)
    bpy.utils.register_class(EXPLORATORY_UL_ProxyMeshList)
    bpy.utils.register_class(EXPLORATORY_OT_AddProxyMesh)
    bpy.utils.register_class(EXPLORATORY_OT_RemoveProxyMesh)
    bpy.types.Scene.character_actions = bpy.props.PointerProperty(type=CharacterActionsPG)

    add_scene_properties()
    print("Exploratory Add-on Registered!")


def unregister():
    remove_scene_properties()
    unregister_interaction_properties()
    unregister_objective_properties()

    #mobility game reactions
    del bpy.types.Scene.mobility_game
    bpy.utils.unregister_class(MobilityGameReactionsPG)

    #reset game
    bpy.utils.unregister_class(EXPLORATORY_OT_ResetGame)


    del bpy.types.Scene.character_audio
    bpy.utils.unregister_class(AUDIO_OT_TestSoundPointer)
    bpy.utils.unregister_class(CharacterAudioPG)
    bpy.utils.unregister_class(EXPLORATORY_OT_BuildAudio)

    bpy.utils.unregister_class(ExpModal)
    bpy.utils.unregister_class(EXP_GAME_OT_StartGame)
    bpy.utils.unregister_class(EXPLORATORY_UL_ReactionsInInteraction)
    bpy.utils.unregister_class(EXPLORATORY_UL_CustomInteractions)
    bpy.utils.unregister_class(VIEW3D_PT_Exploratory_Studio)
    bpy.utils.unregister_class(ExploratoryPanel)
    bpy.utils.unregister_class(VIEW3D_PT_Objectives)
    bpy.utils.unregister_class(EXPLORATORY_UL_Objectives)
    bpy.utils.unregister_class(ExploratoryCharacterPanel)
    bpy.utils.unregister_class(ExploratoryProxyMeshPanel)
    


    bpy.utils.unregister_class(EXPLORATORY_OT_RemoveReactionFromInteraction)
    bpy.utils.unregister_class(EXPLORATORY_OT_AddReactionToInteraction)
    bpy.utils.unregister_class(EXPLORATORY_OT_RemoveInteraction)
    bpy.utils.unregister_class(EXPLORATORY_OT_AddInteraction)

    bpy.utils.unregister_class(InteractionDefinition)
    bpy.utils.unregister_class(ReactionDefinition)

    del bpy.types.Scene.character_actions
    bpy.utils.unregister_class(CharacterActionsPG)
    bpy.utils.unregister_class(ProxyMeshEntry)
    bpy.utils.unregister_class(EXPLORATORY_UL_ProxyMeshList)
    bpy.utils.unregister_class(EXPLORATORY_OT_AddProxyMesh)
    bpy.utils.unregister_class(EXPLORATORY_OT_RemoveProxyMesh)

    print("Exploratory Add-on Unregistered!")

if __name__ == "__main__":
    register()