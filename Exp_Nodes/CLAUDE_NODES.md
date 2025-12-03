# CLAUDENODES.md
This file provides guidance to Claude Code when working with the Exploratory node system.

**📖 REQUIRED READING:** Before working on ANY node system features, read `../Exp_Game/CLAUDE.md` for the complete ENGINE-FIRST architecture. The engine is the foundation that makes Exploratory possible.

---

## 🚨 ULTRA CRITICAL: ENGINE-FIRST MANDATE

**BEFORE implementing ANY feature in the node system or backend:**

1. ✅ **READ `Exp_Game/CLAUDE.md` FIRST** - It contains the ENGINE-FIRST architecture
2. ✅ **ALWAYS ask: "Can this be offloaded to the engine?"**
3. ✅ **NEVER assume work should happen on the main thread**
4. ✅ **The engine (multiprocessing, 4 workers) is the HEART of Exploratory**

### The Golden Rule: Main Thread is ONLY for bpy

```
❌ WRONG - Main thread doing heavy work:
def check_interactions():
    for inter in scene.custom_interactions:
        # Distance checks, raycasts, comparisons...  # TOO HEAVY!

✅ CORRECT - Engine offload pattern:
def check_interactions():
    # 1. Snapshot (pickle-safe data)
    data = {"positions": [...], "distances": [...]}

    # 2. Submit to worker (non-blocking)
    job_id = engine.submit_job("INTERACTION_CHECK", data)

    # 3. Poll results later (in game loop)
    results = engine.poll_results()

    # 4. Apply to Blender (main thread, bpy writes only)
    for result in results:
        fire_interaction(result["interaction_index"])
```

### When Designing New Nodes

**ALWAYS consider:**
- Can trigger checks be offloaded? (distance, collision, state comparisons)
- Can reaction execution be computed in parallel? (transform calculations, pathfinding)
- Can data processing happen in workers? (vector math, graph traversal)

**The engine exists to FREE the main thread for smooth 30Hz modal operation.**

See `Exp_Game/CLAUDE.md` for complete engine architecture, job patterns, and performance guidelines.

---

## ⚠️ CRITICAL: Node System Philosophy

**The node graph is NOT the runtime logic system. It's a VISUAL HELPER that strings together building blocks.**

### Core Principle: Visualization, Not Execution

```
┌─────────────────────────────────────────────────────────────┐
│  NODE GRAPH (Exp_Nodes/)                                     │
│  • Visual design-time editor                                 │
│  • Strings together triggers → reactions                     │
│  • Creates/manages Scene property data                       │
│  • NOT read during gameplay                                  │
└─────────────────────────────────────────────────────────────┘
                        ↓ (writes to)
┌─────────────────────────────────────────────────────────────┐
│  SCENE PROPERTIES (bpy.types.Scene)                          │
│  • scene.custom_interactions (trigger definitions)           │
│  • scene.reactions (reaction definitions)                    │
│  • scene.objectives (objective definitions)                  │
│  • scene.action_keys (action key registry)                   │
└─────────────────────────────────────────────────────────────┘
                        ↓ (read by)
┌─────────────────────────────────────────────────────────────┐
│  RUNTIME GAME LOOP (Exp_Game/)                               │
│  • Reads scene properties ONLY                               │
│  • Executes interactions/reactions                           │
│  • Never touches node graph                                  │
└─────────────────────────────────────────────────────────────┘
```

**Why this matters:**
- Nodes are **design-time tools** for artists/designers
- Game loop reads **scene data structures**, NOT nodes
- This separation allows non-node workflows (direct property editing, procedural generation, etc.)
- Nodes exist to make configuration easier, not to BE the configuration

---

## 🔑 Key Terminology

### Triggers = Interactions

**In the node graph:**
- Called "Trigger Nodes" (ProximityTriggerNode, CollisionTriggerNode, etc.)

**In the runtime:**
- Called "Interactions" (scene.custom_interactions)
- Defined in `Exp_Game/interactions/exp_interaction_definition.py`

**They are the same thing.** The node is just a visual wrapper around an InteractionDefinition.

### Node Graph Structure Rules

**LINEAR CHAINS ONLY - NO BRANCHING:**

```
✅ CORRECT:
[Trigger] → [Reaction A] → [Reaction B] → [Reaction C]

❌ WRONG - NO BRANCHING:
                    ┌→ [Reaction B]
[Trigger] → [Reaction A]
                    └→ [Reaction C]
```

**Why linear?**
- One trigger can execute multiple reactions in sequence
- Branching would create ambiguous execution order
- The InteractionDefinition.reaction_links is an **ordered list** of reaction indices
- Graph traversal reads connections in link order to build this list

---

## 📁 File Structure

```
Exp_Nodes/
├── CLAUDENODES.md              ← YOU ARE HERE
├── __init__.py                 ← Registration (imports all nodes, registers classes)
├── base_nodes.py               ← TriggerNodeBase, ReactionNodeBase (empty base classes)
├── node_editor.py              ← Tree type, sockets, operators, N-panel, Add menus
├── trigger_nodes.py            ← All trigger node types (8 triggers)
├── reaction_nodes.py           ← All reaction node types (17+ reactions)
├── objective_nodes.py          ← Objective node
├── action_key_nodes.py         ← Create Action Key node
├── utility_nodes.py            ← Utility nodes (Delay, Capture Float Vector)
└── trig_react_obj_lists.py     ← N-PANEL LISTS (LEGACY but must stay in sync)
```

---

## 🎯 The N-Panel Legacy System

**Location:** `trig_react_obj_lists.py`

**Purpose:**
- Provides UI lists (UILists) for viewing interactions/reactions in the 3D View N-panel
- Shows "receipts" or "visual proof" of what the node system created
- **LEGACY**: Not used to string together game logic, but must stay updated

**Classes:**
- `VIEW3D_PT_Exploratory_Studio` - Studio panel (interactions list)
- `EXPLORATORY_UL_CustomInteractions` - Interaction list UI
- `EXPLORATORY_UL_ReactionsInInteraction` - Reactions within an interaction
- `EXPLORATORY_UL_ReactionLibrary` - Global reaction library
- `VIEW3D_PT_Exploratory_Reactions` - Reactions panel
- `VIEW3D_PT_Objectives` - Objectives panel
- `EXPLORATORY_UL_Objectives` - Objectives list UI
- Operators: `EXPLORATORY_OT_AddGlobalReaction`, `EXPLORATORY_OT_RemoveGlobalReaction`

**When to Update:**
- ✅ **ALWAYS** when adding a new trigger node type
- ✅ **ALWAYS** when adding a new reaction node type
- ✅ **ALWAYS** when adding a new objective node type
- This keeps the N-panel UI in sync with available node types

---

## 🔨 Adding a New Trigger Node

### Step 1: Define Runtime Interaction (Exp_Game side)

**File:** `Exp_Game/interactions/exp_interaction_definition.py`

```python
# Add to InteractionDefinition.trigger_type enum
trigger_type: bpy.props.EnumProperty(
    items=[
        # ... existing triggers ...
        ("MY_NEW_TRIGGER", "My New Trigger", "Description"),
    ]
)

# Add any trigger-specific properties
my_new_trigger_property: bpy.props.FloatProperty(...)
```

**File:** `Exp_Game/interactions/exp_interactions.py`

```python
# Add trigger check logic to check_interactions()
def check_interactions(scene):
    for inter in scene.custom_interactions:
        if inter.trigger_type == "MY_NEW_TRIGGER":
            if _check_my_new_trigger(inter):
                fire_interaction(inter)
```

### Step 2: Create Node (Exp_Nodes side)

**File:** `Exp_Nodes/trigger_nodes.py`

```python
class MyNewTriggerNode(TriggerNodeBase):
    bl_idname = "MyNewTriggerNodeType"
    bl_label = "My New Trigger"

    interaction_index: bpy.props.IntProperty(default=-1, min=-1)

    def init(self, context):
        self.outputs.new("TriggerOutputSocketType", "Trigger Output")
        self.width = 300
        self._tint()
        self.interaction_index = _ensure_interaction("MY_NEW_TRIGGER")

    def copy(self, node):
        # Duplicate the interaction
        src_idx = getattr(node, "interaction_index", -1)
        # ... duplication logic ...

    def free(self):
        # Clean up interaction when node deleted
        _delete_interaction_at(self.interaction_index)

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.interaction_index
        if not scn or not (0 <= idx < len(scn.custom_interactions)):
            layout.label(text="(Missing Interaction)", icon='ERROR')
            return
        inter = scn.custom_interactions[idx]

        # Draw UI for interaction properties
        layout.prop(inter, "my_new_trigger_property")
```

### Step 3: Register Node

**File:** `Exp_Nodes/__init__.py`

```python
# Import
from .trigger_nodes import (
    # ... existing imports ...
    MyNewTriggerNode,
)

# Add to classes list
classes = [
    # ... existing classes ...
    MyNewTriggerNode,
]
```

### Step 4: Add to Node Editor Menu

**File:** `Exp_Nodes/node_editor.py`

```python
class NODE_MT_exploratory_add_triggers(Menu):
    def draw(self, context):
        layout = self.layout
        # ... existing triggers ...
        add("My New Trigger", "MyNewTriggerNodeType")
```

### Step 5: Update N-Panel Lists (REQUIRED)

**File:** `Exp_Nodes/trig_react_obj_lists.py`

Update the UI lists to show the new trigger type in the N-panel. This is **record-keeping only** but must be done.

---

## 🔨 Adding a New Reaction Node

### Step 1: Define Runtime Reaction (Exp_Game side)

**File:** `Exp_Game/reactions/exp_reaction_definition.py`

```python
# Add to ReactionDefinition.reaction_type enum
reaction_type: bpy.props.EnumProperty(
    items=[
        # ... existing reactions ...
        ("MY_NEW_REACTION", "My New Reaction", "Description"),
    ]
)

# Add reaction-specific properties
my_new_reaction_property: bpy.props.FloatProperty(...)
```

**File:** `Exp_Game/reactions/exp_my_new_reaction.py` (create new file)

```python
# Implement execution logic
def execute_my_new_reaction(r):
    """Execute MY_NEW_REACTION reaction."""
    # Read properties from ReactionDefinition instance 'r'
    value = getattr(r, "my_new_reaction_property", 0.0)
    # ... do the thing ...
```

**File:** `Exp_Game/interactions/exp_interactions.py`

```python
# Import executor
from ..reactions.exp_my_new_reaction import execute_my_new_reaction

# Add to _execute_reaction_now()
def _execute_reaction_now(r):
    t = getattr(r, "reaction_type", "")
    # ... existing reactions ...
    elif t == "MY_NEW_REACTION":
        execute_my_new_reaction(r)
```

### Step 2: Create Node (Exp_Nodes side)

**File:** `Exp_Nodes/reaction_nodes.py`

```python
class ReactionMyNewReactionNode(_ReactionNodeKind):
    bl_idname = "ReactionMyNewReactionNodeType"
    bl_label = "My New Reaction"
    KIND = "MY_NEW_REACTION"

    def draw_buttons(self, context, layout):
        scn = _scene()
        idx = self.reaction_index
        if not scn or not (0 <= idx < len(scn.reactions)):
            layout.label(text="(Missing Reaction)", icon='ERROR')
            return
        r = scn.reactions[idx]

        box = layout.box()
        box.prop(r, "name", text="Name")

        # Draw UI for reaction properties
        layout.prop(r, "my_new_reaction_property")
```

**Note:** By inheriting from `_ReactionNodeKind`, the node automatically gets:
- `init()` - creates a ReactionDefinition in scene.reactions
- `copy()` - duplicates the reaction
- `free()` - cleans up and reindexes
- Input/output sockets for chaining

### Step 3: Register Node

**File:** `Exp_Nodes/__init__.py`

```python
# Import
from .reaction_nodes import (
    # ... existing imports ...
    ReactionMyNewReactionNode,
)

# Add to classes list
classes = [
    # ... existing classes ...
    ReactionMyNewReactionNode,
]
```

### Step 4: Add to Node Editor Menu

**File:** `Exp_Nodes/node_editor.py`

```python
class NODE_MT_exploratory_add_reactions(Menu):
    def draw(self, context):
        layout = self.layout
        # ... existing reactions ...
        add("My New Reaction", "ReactionMyNewReactionNodeType")
```

### Step 5: Update N-Panel Lists (REQUIRED)

**File:** `Exp_Nodes/trig_react_obj_lists.py`

Update the UI lists to show the new reaction type in the N-panel. This is **record-keeping only** but must be done.

---

## 🔗 Node → Scene Data Flow

### Trigger Nodes

**On creation (`init()`):**
```python
self.interaction_index = _ensure_interaction("PROXIMITY")
```
This creates a new entry in `scene.custom_interactions` and stores its index.

**On connection change:**
```python
_sync_interaction_links_from_graph(trigger_node, inter_index)
```
This reads the connected reaction chain and writes to `InteractionDefinition.reaction_links`.

**On deletion (`free()`):**
```python
_delete_interaction_at(self.interaction_index)
```
Removes the interaction from the scene and reindexes all surviving nodes.

### Reaction Nodes

**On creation (`init()`):**
```python
self.reaction_index = _ensure_reaction(self.KIND)
```
This creates a new entry in `scene.reactions` and stores its index.

**Node properties → Reaction properties:**
The node's `draw_buttons()` method exposes `r.property_name` directly, so changes write immediately to the scene data.

**On deletion (`free()`):**
```python
scn.reactions.remove(idx)
_fix_interaction_reaction_indices_after_remove(idx)
_reindex_reaction_nodes_after_remove(idx)
```
Removes the reaction, fixes all interaction links, and reindexes all nodes.

---

## 🎮 Runtime Execution (Game Loop)

**File:** `Exp_Game/modal/exp_loop.py`

```python
# Every frame:
check_interactions(context.scene)  # Check all triggers

# On trigger fire:
def fire_interaction(inter):
    # Get ordered list of reactions
    reactions = [scene.reactions[link.reaction_index]
                 for link in inter.reaction_links]

    # Execute each reaction in sequence
    for r in reactions:
        _execute_reaction_now(r)
```

**The game loop NEVER:**
- ❌ Reads from the node graph
- ❌ Traverses node connections
- ❌ Looks at node properties

**The game loop ONLY:**
- ✅ Reads `scene.custom_interactions`
- ✅ Reads `scene.reactions`
- ✅ Executes based on scene data

---

## 🧩 Special Node Types

### Utility Nodes

**Examples:** Delay, Capture Float Vector

- Inherit from `_ReactionNodeKind` (have `reaction_index`)
- Appear in reaction chains
- Some affect execution timing (Delay), others capture/store data

### Objective Nodes

**File:** `Exp_Nodes/objective_nodes.py`

- Standalone nodes (not in trigger/reaction chains)
- Create entries in `scene.objectives`
- Used by trigger nodes (Objective Update, Timer Complete)
- Used by reaction nodes (Objective Counter, Objective Timer, Custom UI Text with objective display)

### Action Key Nodes

**File:** `Exp_Nodes/action_key_nodes.py`

- Create entries in `scene.action_keys` (global registry)
- Referenced by Action Trigger nodes (by name)
- Referenced by Action Keys Reaction nodes (by name)

---

## ⚠️ Common Pitfalls

### ❌ DON'T: Read node graph at runtime
```python
# WRONG - runtime code should NEVER do this
for node in node_tree.nodes:
    if node.bl_idname == "ProximityTriggerNode":
        # ... NO! ...
```

### ✅ DO: Read scene properties at runtime
```python
# CORRECT
for inter in scene.custom_interactions:
    if inter.trigger_type == "PROXIMITY":
        # ... YES! ...
```

### ❌ DON'T: Create branching node graphs
```
# WRONG
[Trigger] → [Reaction A] → [Reaction B]
                         ↘ [Reaction C]  # Ambiguous order!
```

### ✅ DO: Create linear chains
```
# CORRECT
[Trigger] → [Reaction A] → [Reaction B] → [Reaction C]
```

### ❌ DON'T: Skip updating N-panel lists
When you add a new node type, the N-panel lists must be updated even though they're legacy. They provide visual confirmation and debugging value.

### ✅ DO: Keep N-panel in sync
Update `trig_react_obj_lists.py` every time you add/modify node types.

---

## 🔍 Debugging Node Issues

### Node not appearing in Add menu?

1. Check `__init__.py` - is it imported?
2. Check `__init__.py` - is it in the `classes` list?
3. Check `node_editor.py` - is it in the appropriate `NODE_MT_exploratory_add_*` menu?

### Node creates but doesn't execute at runtime?

1. Check `exp_reaction_definition.py` - is the reaction type in the enum?
2. Check `exp_interactions.py` - is the reaction type handled in `_execute_reaction_now()`?
3. Check the executor function - is it being called correctly?

### Node deletes but leaves orphan data?

1. Check `free()` method - does it call the appropriate cleanup function?
2. Check reindexing - are other nodes updating their indices correctly?

### Properties not saving?

Remember: Node properties are **ephemeral**. Only scene properties (via `scene.custom_interactions`, `scene.reactions`, etc.) persist. Nodes should expose scene properties directly in `draw_buttons()`.

---

## 📋 Checklist: Adding a New Reaction Node

- [ ] Add reaction type to `ReactionDefinition.reaction_type` enum
- [ ] Add reaction properties to `ReactionDefinition`
- [ ] Create executor function in `Exp_Game/reactions/`
- [ ] Add executor call to `_execute_reaction_now()` in `exp_interactions.py`
- [ ] Create node class in `reaction_nodes.py` (inherit from `_ReactionNodeKind`)
- [ ] Import node in `Exp_Nodes/__init__.py`
- [ ] Add node to `classes` list in `__init__.py`
- [ ] Add node to `NODE_MT_exploratory_add_reactions` menu in `node_editor.py`
- [ ] Update N-panel lists in `trig_react_obj_lists.py`
- [ ] Test: Create node, connect to trigger, verify execution at runtime
- [ ] Test: Delete node, verify cleanup and reindexing

---

## 📋 Checklist: Adding a New Trigger Node

- [ ] Add trigger type to `InteractionDefinition.trigger_type` enum
- [ ] Add trigger properties to `InteractionDefinition`
- [ ] Add trigger check logic to `check_interactions()` in `exp_interactions.py`
- [ ] Create node class in `trigger_nodes.py` (inherit from `TriggerNodeBase`)
- [ ] Implement `init()`, `copy()`, `free()`, and `draw_buttons()`
- [ ] Import node in `Exp_Nodes/__init__.py`
- [ ] Add node to `classes` list in `__init__.py`
- [ ] Add node to `NODE_MT_exploratory_add_triggers` menu in `node_editor.py`
- [ ] Update N-panel lists in `trig_react_obj_lists.py`
- [ ] Test: Create node, connect reactions, verify trigger fires at runtime
- [ ] Test: Delete node, verify cleanup and reindexing

---

## 🏗️ Architecture Summary

```
Design Time:              Runtime:

Node Graph Editor         Game Loop (30Hz)
     ↓                         ↑
Nodes create/edit             Reads only
     ↓                         ↑
Scene Properties ←────────────┘
(custom_interactions,
 reactions,
 objectives,
 action_keys)
```

**Key insight:** Nodes are a **convenience layer** for editing scene properties. They are not the source of truth. The scene properties are the source of truth. This allows:

1. Non-node workflows (direct property editing, Python API, procedural generation)
2. Node graph independence (can delete entire node tree, data remains)
3. Performance (runtime doesn't parse node connections)
4. Flexibility (can add nodes retroactively to existing data)

---

**Remember:** The nodes are the **paintbrush**, not the **painting**. They create the scene data, but the game reads the scene data directly.

---

## 🔮 Advanced: Data Passing with Utility Nodes

### The Capture Float Vector System

**Status:** Experimental, powerful concept, but **backend is not fully ready** for this paradigm.

### Vision: Real-Time Data Flow Between Nodes

The Capture Float Vector node represents a **powerful architectural concept**: passing runtime data (vectors, booleans, floats) between nodes in real-time during gameplay.

**Current Implementation:**
- `UtilityCaptureFloatVectorNode` - Stores a 3-float vector with timestamp
- `exp_utility_store.py` - Scene-level registry (UID-based storage)
- Can capture impact locations, player positions, dynamic values
- Can pass data to other nodes via socket connections

**Files:**
- `Exp_Nodes/utility_nodes.py` - Node definition
- `Exp_Game/props_and_utils/exp_utility_store.py` - Storage backend

### How Capture Float Vector Works (Currently)

```
1. Node Creation:
   - Creates a unique UID for this capture slot
   - Registers in scene.utility_float_vectors

2. Runtime Write:
   - Reaction executor calls node.write_from_graph(vec3)
   - Stores vector + timestamp in scene registry
   - Syncs to any connected downstream nodes

3. Runtime Read:
   - Other nodes can read via get_floatvec(uid)
   - Supports chaining (Capture → Capture)
   - Supports export_vector() contract for custom providers
```

### Current Use Cases

**Projectile/Hitscan Impact Locations:**
- Hitscan reaction fires → stores impact point in Capture node
- Transform reaction can read that location
- Can move objects to impact point

**Example Flow:**
```
[Hitscan Reaction] → writes impact location
                    ↓
[Capture Float Vector] → stores (x, y, z)
                    ↓
[Transform Reaction] → reads location, moves object
```

### The Problem: Backend Not Ready for Full Data Flow

**What's missing:**

1. **No Real-Time Polling Pattern**
   - Current system writes ONCE per trigger fire
   - No continuous monitoring of changing values
   - Can't react to "when vector changes" or "when value exceeds threshold"

2. **Limited Data Types**
   - Only Float Vectors (3D) implemented
   - No Boolean sockets (for conditional logic)
   - No Float sockets (for scalar values)
   - No String sockets (for state labels)

3. **No Worker Offload for Data Processing**
   - Vector math happens on main thread
   - Comparisons happen on main thread
   - Should be offloaded to engine workers for performance

4. **No Trigger Integration**
   - Can't create triggers based on stored values
   - No "Value Changed" trigger
   - No "Threshold Exceeded" trigger
   - No "Distance Between Vectors" trigger

### Future Vision: Engine-Offloaded Data Flow

**What this SHOULD become:**

```
┌─────────────────────────────────────────────────────────────┐
│  DESIGN TIME: Node Graph                                     │
│  [Capture Position A] ──→ [Distance Check] ──→ [Trigger]   │
│         ↓                        ↓                           │
│  Registers slot A         Registers comparison job          │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  RUNTIME: Engine Workers (30Hz)                              │
│  - Read slot A, slot B positions                             │
│  - Compute distance (worker, no bpy)                         │
│  - Check threshold                                           │
│  - Return trigger decision                                   │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  MAIN THREAD: Apply Results                                  │
│  - Fire trigger if threshold met                             │
│  - Execute reactions                                         │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles for Data Flow Nodes (Future)

When implementing data-passing features:

1. **Storage Must Be Scene-Based**
   - ✅ Store in scene properties (scene.utility_*)
   - ❌ Don't store in node properties (ephemeral)
   - UID-based lookup for stability

2. **Computation Must Be Engine-Offloaded**
   - ✅ Vector math → worker
   - ✅ Distance checks → worker
   - ✅ Comparisons → worker
   - ❌ Don't compute on main thread

3. **Reads Must Be Non-Blocking**
   - ✅ Poll pattern (check if value ready)
   - ✅ Cached results
   - ❌ Don't wait/block for values

4. **Writes Must Be Timestamped**
   - ✅ Track when value was updated
   - ✅ Detect stale data
   - ✅ Support interpolation/history

### Adding New Data Socket Types (Template)

**Step 1: Define Socket Classes**

```python
# Exp_Nodes/utility_nodes.py
class BooleanInputSocket(bpy.types.NodeSocket):
    bl_idname = "BooleanInputSocketType"
    bl_label = "Boolean (In)"

class BooleanOutputSocket(bpy.types.NodeSocket):
    bl_idname = "BooleanOutputSocketType"
    bl_label = "Boolean (Out)"
```

**Step 2: Define Storage PropertyGroup**

```python
# Exp_Game/props_and_utils/exp_utility_store.py
class UtilityBooleanSlotPG(PropertyGroup):
    uid: bpy.props.StringProperty()
    has_value: bpy.props.BoolProperty(default=False)
    value: bpy.props.BoolProperty(default=False)
    updated_at: bpy.props.FloatProperty(default=0.0)
```

**Step 3: Create Capture Node**

```python
# Exp_Nodes/utility_nodes.py
class UtilityCaptureBooleanNode(_ExploratoryNodeOnly, Node):
    bl_idname = "UtilityCaptureBooleanNodeType"
    bl_label = "Capture Boolean"

    capture_uid: bpy.props.StringProperty()

    def init(self, context):
        self.inputs.new("BooleanInputSocketType", "Bool In")
        self.outputs.new("BooleanOutputSocketType", "Bool Out")
        self.capture_uid = create_boolean_slot(context.scene)
```

**Step 4: (CRITICAL) Design Engine Offload Pattern**

```python
# Exp_Game/interactions/exp_interactions.py
def check_boolean_triggers():
    # 1. Snapshot (main thread)
    data = {
        "slots": [(uid, last_value) for uid in active_boolean_slots]
    }

    # 2. Submit to worker
    job_id = engine.submit_job("BOOLEAN_CHECK", data)

    # 3. Poll results (later in loop)
    # 4. Fire triggers based on worker results
```

### Current Limitations Summary

**Capture Float Vector is a proof-of-concept, but:**

❌ Backend not designed for real-time data passing
❌ No worker offload for vector operations
❌ Limited to one data type (3D vectors)
❌ No trigger integration
❌ No continuous monitoring

**To expand this system:**

1. ✅ Read `Exp_Game/CLAUDE.md` for engine patterns
2. ✅ Design worker job types for data operations
3. ✅ Create scene-based storage for new data types
4. ✅ Implement poll-based value reading (non-blocking)
5. ✅ Add trigger nodes that consume utility data

### Key Insight: Don't Fight the Architecture

The current system is **interaction-driven** (triggers fire → reactions execute).

Adding real-time data flow requires **extending this model** with:
- Continuous monitoring (30Hz checks)
- Engine-offloaded comparisons
- Scene-based value storage
- Non-blocking reads

**This is a major architectural expansion, not a small feature add.**

---

## 🎓 Learning from Capture Float Vector

The Capture Float Vector node teaches us:

1. ✅ **Scene-based storage works** - UID registry is stable
2. ✅ **Socket connections work** - Can chain nodes visually
3. ✅ **Write-once pattern works** - Store impact location, read later
4. ❌ **Main thread computation doesn't scale** - Need engine offload
5. ❌ **No real-time monitoring** - Need continuous worker checks

**Use this as a template for future utility nodes, but ALWAYS ask: "How does this offload to the engine?"**

---
