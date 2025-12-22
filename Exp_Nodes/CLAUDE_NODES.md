# CLAUDE_NODES.md

This file provides guidance to Claude Code when working with the Exploratory node system.

**REQUIRED READING:** Before working on ANY node system features, read `../Exp_Game/CLAUDE.md` for the complete ENGINE-FIRST architecture.

---

## ⚠️ CRITICAL: Socket Layout Requirements ⚠️

**SOCKETS MUST ALWAYS BE INLINE WITH THEIR DATA FIELDS. NEVER SEPARATE THEM.**

```
❌ WRONG - Sockets at bottom, fields at top:
┌─────────────────────────┐
│  [Object Picker]        │
│  [Object Picker]        │
│  [Operator] [Distance]  │
│  Hz: 10                 │
│                         │
│  ● Object A             │  ← TERRIBLE! Sockets disconnected from fields
│  ● Object B             │
│  ● Distance             │
└─────────────────────────┘

✅ CORRECT - Sockets inline with their fields:
┌─────────────────────────┐
│  ● [Object Picker]      │  ← Socket inline with its data field
│  ● [Object Picker]      │  ← Socket inline with its data field
│  [Operator] ● [Distance]│  ← Socket inline with its data field
│  Hz: 10                 │
└─────────────────────────┘
```

**HOW TO IMPLEMENT:**
- Do NOT create input sockets in `init()` and then draw fields separately in `draw_buttons()`
- Use the socket's `draw()` method to render the property inline when not connected
- Store `prop_name` on sockets to know which node property to draw
- When socket is linked, show only the socket dot and label
- When socket is NOT linked, draw the property picker/field inline

**This applies to ALL nodes with connectable inputs. No exceptions.**

---

## CRITICAL: Architecture Principles

### 1. Node Graph = Design-Time Configuration ONLY

```
┌─────────────────────────────────────────────────────────────┐
│  NODE GRAPH (Exp_Nodes/)                                     │
│  • Visual design-time editor                                 │
│  • Configures triggers, reactions, trackers, conditions      │
│  • Writes to scene properties                                │
│  • NEVER evaluated at runtime                                │
│  • NEVER traversed during gameplay                           │
└─────────────────────────────────────────────────────────────┘
                        ↓ writes configuration to
┌─────────────────────────────────────────────────────────────┐
│  SCENE PROPERTIES (bpy.types.Scene)                          │
│  • scene.custom_interactions (trigger definitions)           │
│  • scene.reactions (reaction definitions)                    │
│  • scene.trackers (NEW: runtime state tracking)              │
│  • scene.conditions (NEW: conditional logic)                 │
│  • Bindings: which parameters use dynamic values             │
└─────────────────────────────────────────────────────────────┘
                        ↓ snapshot sent to
┌─────────────────────────────────────────────────────────────┐
│  ENGINE WORKER (multiprocessing)                             │
│  • Updates all trackers every frame                          │
│  • Evaluates all conditions                                  │
│  • Resolves dynamic parameter bindings                       │
│  • Returns: which triggers to fire, resolved values          │
│  • NO bpy access - pure Python computation                   │
└─────────────────────────────────────────────────────────────┘
                        ↓ results applied by
┌─────────────────────────────────────────────────────────────┐
│  MAIN THREAD (30Hz modal)                                    │
│  • Receives worker results                                   │
│  • Fires triggers with resolved values                       │
│  • Executes reactions (bpy writes only)                      │
│  • Snapshots new state for next frame                        │
└─────────────────────────────────────────────────────────────┘
```

### 2. Engine Worker Does ALL Computation

**The engine worker (in `/Exp_Game/engine/`) MUST handle:**
- Tracker value updates (reading object positions, speeds, states)
- Condition evaluation (comparisons, AND/OR logic)
- Dynamic parameter resolution (looking up bound values)
- Distance calculations, vector math, threshold checks

**The main thread ONLY does:**
- bpy property reads (snapshotting)
- bpy property writes (applying results)
- Firing reactions

### 3. Performance is Non-Negotiable

```
❌ NEVER: Evaluate node graph connections at runtime
❌ NEVER: Traverse nodes during gameplay
❌ NEVER: Do math on main thread that could be in worker
❌ NEVER: Poll or check values on main thread

✅ ALWAYS: Pre-compute configuration at design time
✅ ALWAYS: Snapshot state, send to worker, poll results
✅ ALWAYS: Offload comparisons/math to worker
✅ ALWAYS: Use non-blocking patterns
```

---

## THE NEW ARCHITECTURE: Trackers + Data Nodes + Dynamic Inputs

### Overview

The system has three new concepts:

1. **Trackers** - Named data streams updated every frame in the engine worker
2. **Data Nodes** - Visual nodes that define values (hardcoded or from trackers)
3. **Dynamic Inputs** - Interaction/Reaction parameters that can be bound to data sources

### Trackers (Engine-Side State Tracking)

A **Tracker** is a named lens into any piece of runtime data. Trackers are:
- Defined at design-time via Tracker Nodes
- Updated every frame in the engine worker
- Available for conditions and dynamic parameter binding

**Tracker Categories:**

| Category | Source Type | Examples |
|----------|-------------|----------|
| **Object Property** | `OBJECT_PROPERTY` | `door.location.z`, `light.energy`, `enemy["health"]` |
| **Object Relationship** | `DISTANCE`, `ANGLE`, `CONTACT` | `distance(player, goal)`, `is_touching(player, lava)` |
| **Character State** | `CHARACTER_*` | `CHARACTER_SPEED`, `CHARACTER_GROUNDED`, `CHARACTER_SPRINTING` |
| **World/System** | `GAME_TIME`, `OBJECTIVE_VALUE` | `game_time`, `objective["score"].value` |
| **Computed** | `EXPRESSION` | `tracker_a - tracker_b`, `tracker_x * 2.0` |

**Tracker Definition (Scene Property):**

```python
class TrackerDefinition(PropertyGroup):
    name: StringProperty()                    # "player_speed"
    tracker_type: EnumProperty(...)           # OBJECT_PROPERTY, DISTANCE, CHARACTER_SPEED, etc.
    data_type: EnumProperty(...)              # FLOAT, INT, BOOL, VECTOR, OBJECT

    # Source configuration (varies by tracker_type)
    source_object: PointerProperty(...)       # For OBJECT_PROPERTY
    property_path: StringProperty()           # For OBJECT_PROPERTY
    object_a: PointerProperty(...)            # For DISTANCE, CONTACT
    object_b: PointerProperty(...)            # For DISTANCE, CONTACT

    # Runtime state (updated by engine worker)
    current_value_float: FloatProperty()
    current_value_int: IntProperty()
    current_value_bool: BoolProperty()
    current_value_vector: FloatVectorProperty()
```

**Engine Worker Updates Trackers:**

```python
# In engine worker (every frame)
def update_trackers(snapshot):
    results = {}
    for tracker in snapshot["trackers"]:
        if tracker["type"] == "OBJECT_PROPERTY":
            # Read from snapshot (no bpy access in worker!)
            obj_data = snapshot["objects"][tracker["object_name"]]
            value = resolve_property_path(obj_data, tracker["property_path"])
            results[tracker["name"]] = value

        elif tracker["type"] == "DISTANCE":
            pos_a = snapshot["objects"][tracker["object_a"]]["location"]
            pos_b = snapshot["objects"][tracker["object_b"]]["location"]
            results[tracker["name"]] = distance(pos_a, pos_b)

        elif tracker["type"] == "CHARACTER_SPEED":
            results[tracker["name"]] = snapshot["character"]["speed"]

    return results
```

### Data Nodes (Design-Time Value Sources)

**Data Nodes** produce typed values that can be wired into interaction/reaction parameters.

| Node Type | Output Type | Value Source |
|-----------|-------------|--------------|
| `FloatNode` | float | Hardcoded value OR bound to float tracker |
| `IntNode` | int | Hardcoded value OR bound to int tracker |
| `BoolNode` | bool | Hardcoded value OR bound to bool tracker |
| `VectorNode` | (x,y,z) | Hardcoded value OR bound to vector tracker |
| `ObjectNode` | object ref | Hardcoded object OR bound to object tracker |
| `TrackerNode` | any (typed) | Defines a new tracker, outputs its value |
| `ConditionNode` | bool | Compares values, outputs true/false |

**Data Node Sockets:**

```python
# Socket types for data flow
class FloatInputSocket(NodeSocket): ...
class FloatOutputSocket(NodeSocket): ...
class IntInputSocket(NodeSocket): ...
class IntOutputSocket(NodeSocket): ...
class BoolInputSocket(NodeSocket): ...
class BoolOutputSocket(NodeSocket): ...
class VectorInputSocket(NodeSocket): ...
class VectorOutputSocket(NodeSocket): ...
class ObjectInputSocket(NodeSocket): ...
class ObjectOutputSocket(NodeSocket): ...
```

### Dynamic Inputs (Bindable Parameters)

Every parameter on every interaction/reaction can be either:
- **Hardcoded** - User types in a value, stored directly
- **Bound** - Linked to a data source (tracker, computed value, etc.)

**Example: Proximity Trigger with Dynamic Distance**

```
┌─────────────────────────────────────────────────────────────┐
│  Tracker Node                                                │
│  Name: "alert_range"                                         │
│  Type: OBJECT_PROPERTY                                       │
│  Object: AlertRangeSphere                                    │
│  Property: scale.x                                           │
│  └─ [Float Output] ─────────────────┐                       │
└─────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────┐
│  Proximity Trigger Node                                      │
│  Object A: [Character]                                       │
│  Object B: Enemy                                             │
│  Distance: [Float Input] ←──────────┘ (bound to tracker)    │
│  └─ [Trigger Output] → ...                                  │
└─────────────────────────────────────────────────────────────┘
```

**Binding Storage:**

```python
class ParameterBinding(PropertyGroup):
    parameter_name: StringProperty()      # "proximity_distance"
    binding_type: EnumProperty(...)       # HARDCODED, TRACKER, COMPUTED
    tracker_name: StringProperty()        # If bound to tracker

class InteractionDefinition(PropertyGroup):
    # ... existing fields ...

    # NEW: Parameter bindings
    bindings: CollectionProperty(type=ParameterBinding)

    # Existing hardcoded values remain for when binding_type == HARDCODED
    proximity_distance: FloatProperty()
```

**Resolution at Runtime (Engine Worker):**

```python
def resolve_interaction_parameters(interaction, tracker_values):
    resolved = {}

    for binding in interaction["bindings"]:
        param = binding["parameter_name"]

        if binding["type"] == "HARDCODED":
            resolved[param] = interaction[param]  # Use stored value

        elif binding["type"] == "TRACKER":
            resolved[param] = tracker_values[binding["tracker_name"]]

    return resolved
```

---

## CONDITIONS: Logical Evaluation

Conditions allow creators to define complex trigger logic:

```
IF character_speed >= 5.0 AND wall_contact == True
THEN fire trigger
```

**Condition Definition:**

```python
class ConditionDefinition(PropertyGroup):
    name: StringProperty()

    # Left side
    left_source: EnumProperty(...)        # TRACKER, LITERAL
    left_tracker: StringProperty()         # If source is TRACKER
    left_value_float: FloatProperty()      # If source is LITERAL

    # Comparison
    operator: EnumProperty(...)            # EQ, NE, LT, LE, GT, GE

    # Right side
    right_source: EnumProperty(...)
    right_tracker: StringProperty()
    right_value_float: FloatProperty()

    # Combination (for compound conditions)
    combinator: EnumProperty(...)          # NONE, AND, OR
    next_condition: IntProperty()          # Index of chained condition
```

**Condition Node:**

```
┌─────────────────────────────────────────────────────────────┐
│  Condition Node                                              │
│  ├─ [Float Input] Left ←── Tracker or Float Node            │
│  ├─ Operator: >= (dropdown)                                  │
│  ├─ [Float Input] Right ←── Tracker, Float Node, or literal │
│  └─ [Bool Output] ──→ To Condition Trigger or AND/OR node  │
└─────────────────────────────────────────────────────────────┘
```

**Condition Trigger Node:**

A new trigger type that fires when a condition becomes true:

```
┌─────────────────────────────────────────────────────────────┐
│  Condition Trigger Node                                      │
│  ├─ [Bool Input] Condition ←── Condition Node output        │
│  ├─ Mode: ON_BECOME_TRUE / WHILE_TRUE / ON_CHANGE           │
│  └─ [Trigger Output] → Reactions...                          │
└─────────────────────────────────────────────────────────────┘
```

**Engine Worker Evaluates Conditions:**

```python
def evaluate_conditions(conditions, tracker_values, previous_states):
    results = []

    for cond in conditions:
        left = resolve_value(cond["left"], tracker_values)
        right = resolve_value(cond["right"], tracker_values)

        current = compare(left, cond["operator"], right)
        previous = previous_states.get(cond["name"], False)

        # Edge detection
        if cond["mode"] == "ON_BECOME_TRUE":
            if current and not previous:
                results.append(cond["trigger_index"])

        elif cond["mode"] == "WHILE_TRUE":
            if current:
                results.append(cond["trigger_index"])

    return results, {c["name"]: current for c in conditions}
```

---

## EXAMPLE: Complete Data Flow

**User Goal:** "If character is sprinting AND touches a wall, make them fall and set speed to 0"

**Node Graph (Design-Time):**

```
┌─────────────────┐     ┌─────────────────┐
│ Tracker Node    │     │ Tracker Node    │
│ "char_speed"    │     │ "wall_contact"  │
│ CHARACTER_SPEED │     │ CONTACT         │
│ [Float Out]─────│──┐  │ [Bool Out]──────│──┐
└─────────────────┘  │  └─────────────────┘  │
                     │                        │
┌────────────────────│────────────────────────│───────────────┐
│ Condition Node     │                        │               │
│ Left: ←────────────┘                        │               │
│ Operator: >=                                │               │
│ Right: 5.0 (hardcoded sprint threshold)     │               │
│ [Bool Out]────────────────────────────────┐ │               │
└───────────────────────────────────────────│─│───────────────┘
                                            │ │
┌───────────────────────────────────────────│─│───────────────┐
│ AND Node                                  │ │               │
│ [Bool In A] ←─────────────────────────────┘ │               │
│ [Bool In B] ←───────────────────────────────┘               │
│ [Bool Out]──────────────────────────────────────────────┐   │
└─────────────────────────────────────────────────────────│───┘
                                                          │
┌─────────────────────────────────────────────────────────│───┐
│ Condition Trigger Node                                  │   │
│ [Bool In] ←─────────────────────────────────────────────┘   │
│ Mode: ON_BECOME_TRUE                                        │
│ [Trigger Out]───→ [Fall Reaction] ───→ [Set Speed Reaction] │
└─────────────────────────────────────────────────────────────┘
```

**Scene Properties (Written at Design-Time):**

```python
scene.trackers = [
    {"name": "char_speed", "type": "CHARACTER_SPEED", "data_type": "FLOAT"},
    {"name": "wall_contact", "type": "CONTACT", "object_a": character, "object_b_tag": "wall", "data_type": "BOOL"},
]

scene.conditions = [
    {"name": "is_sprinting", "left_tracker": "char_speed", "op": "GE", "right_value": 5.0},
]

scene.custom_interactions[X] = {
    "trigger_type": "CONDITION",
    "condition_mode": "ON_BECOME_TRUE",
    "condition_expression": "is_sprinting AND wall_contact",
    "reaction_links": [fall_reaction_idx, set_speed_reaction_idx],
}

scene.reactions[set_speed_idx] = {
    "reaction_type": "SET_TRACKER",
    "target_tracker": "char_speed",
    "value": 0.0,
}
```

**Engine Worker (Every Frame):**

```python
def engine_frame(snapshot):
    # 1. Update all trackers
    tracker_values = update_trackers(snapshot)
    # tracker_values = {"char_speed": 7.2, "wall_contact": True}

    # 2. Evaluate all conditions
    condition_results = evaluate_conditions(snapshot["conditions"], tracker_values)
    # condition_results = {"is_sprinting": True}

    # 3. Evaluate compound expressions
    expression_results = evaluate_expressions(snapshot["expressions"], condition_results, tracker_values)
    # "is_sprinting AND wall_contact" = True AND True = True

    # 4. Check for edge transitions (became true this frame)
    triggers_to_fire = check_edge_triggers(expression_results, previous_states)
    # triggers_to_fire = [interaction_X]

    # 5. Resolve dynamic parameters for triggered interactions
    resolved_params = {}
    for trigger in triggers_to_fire:
        resolved_params[trigger] = resolve_parameters(trigger, tracker_values)

    return {
        "triggers": triggers_to_fire,
        "resolved_params": resolved_params,
        "tracker_values": tracker_values,
    }
```

**Main Thread (Applies Results):**

```python
def apply_engine_results(results):
    for trigger_idx in results["triggers"]:
        interaction = scene.custom_interactions[trigger_idx]
        params = results["resolved_params"][trigger_idx]

        fire_interaction(interaction, params)
```

---

## FILE STRUCTURE

```
Exp_Nodes/
├── CLAUDE_NODES.md              ← YOU ARE HERE
├── __init__.py                  ← Registration
├── base_nodes.py                ← Base classes
├── node_editor.py               ← Tree, sockets, menus, panel
│
├── trigger_nodes.py             ← Trigger nodes (Proximity, Collision, etc.)
├── reaction_nodes.py            ← Reaction nodes (Transform, Sound, etc.)
├── objective_nodes.py           ← Objective node
├── action_key_nodes.py          ← Action Key node
│
├── data_nodes.py                ← NEW: Float, Int, Bool, Vector, Object nodes
├── tracker_nodes.py             ← NEW: Tracker definition nodes
├── condition_nodes.py           ← NEW: Condition, AND, OR, NOT nodes
│
├── utility_nodes.py             ← Utility nodes (Delay, legacy Capture)
└── trig_react_obj_lists.py      ← N-Panel debug visualization
```

---

## N-PANEL DEBUG VISUALIZATION

**File:** `trig_react_obj_lists.py`

The N-Panel provides read-only debugging views:
- Lists all interactions with their chained reactions
- Lists all reactions in the library
- Lists all objectives
- **Detects orphans** - items with no node reference
- **Detects invalid links** - broken reaction indices

**This is for debugging only.** The panels don't edit anything - they just show what exists and flag problems.

---

## CURRENT IMPLEMENTATION: Worker-Offloaded Trackers

### Backend Data Flow (IMPLEMENTED)

The tracker system is now fully worker-offloaded. Here's how it works:

#### 1. Game Start: Serialization

**File:** `Exp_Game/interactions/exp_tracker_eval.py`

At game start, `cache_trackers_in_worker()` is called from `GameLoop.__init__()`:

```python
# Called once at game start
serialize_tracker_graph(scene) → returns list of tracker definitions

# Each tracker definition contains:
{
    'interaction_index': int,      # Which ExternalTriggerNode this feeds
    'trigger_node': str,           # Node name for debugging
    'tree_name': str,              # Node tree name
    'condition_tree': {            # Recursively serialized node tree
        'type': 'DistanceTrackerNodeType',
        'object_a': 'Player',
        'object_b': 'Goal',
        'op': 'LT',
        'value': 5.0,
        'eval_hz': 10,
        'inputs': [...]            # For logic gates (AND/OR/NOT)
    }
}
```

#### 2. Worker Cache: CACHE_TRACKERS Job

**File:** `Exp_Game/engine/worker/entry.py`

Worker receives and caches tracker definitions:

```python
# Worker-side caches (global in entry.py)
_cached_trackers = []           # Serialized tracker definitions
_tracker_states = {}            # {inter_idx: last_bool} for edge detection
_tracker_last_eval = {}         # {inter_idx: last_eval_time} for Hz throttling
```

#### 3. Per-Frame: World State Collection

**File:** `Exp_Game/interactions/exp_tracker_eval.py`

Each frame, `collect_world_state()` gathers minimal data for evaluation:

```python
{
    'positions': {'Player': (x,y,z), 'Enemy': (x,y,z), ...},
    'char_state': 'RUNNING' | 'IDLE' | 'JUMPING' | ...,
    'inputs': {'FORWARD': True, 'JUMP': False, ...},
    'game_time': 12.5,
    'contacts': {'Player': ['Ground', 'Wall'], ...}
}
```

#### 4. Per-Frame: EVALUATE_TRACKERS Job

**File:** `Exp_Game/engine/worker/entry.py`

Worker evaluates all cached trackers with current world state:

```python
def _handle_evaluate_trackers(job_data):
    for tracker in _cached_trackers:
        # Hz throttling
        if game_time - last_eval < eval_interval:
            continue

        # Recursive condition tree evaluation
        new_value = _eval_condition_tree(tree, world_state)

        # Edge detection
        if new_value != old_value:
            signal_updates[inter_idx] = new_value

    return {
        'signal_updates': {0: True, 3: False, ...},
        'fired_indices': [0],
        'calc_time_us': 45.0
    }
```

#### 5. Main Thread: Apply Results

**File:** `Exp_Game/modal/exp_loop.py`

Results are polled and applied:

```python
def _poll_tracker_results(self):
    results = op.engine.poll_results(max_results=10)
    for result in results:
        if process_tracker_result(op, result):
            # Sets inter.external_signal based on signal_updates
            continue
```

Then `check_interactions()` → `handle_external_trigger()` fires when `external_signal` is True.

### Worker-Side Condition Evaluation

**File:** `Exp_Game/engine/worker/entry.py`

The `_eval_condition_tree()` function recursively evaluates:

| Node Type | Evaluation |
|-----------|------------|
| `DistanceTrackerNodeType` | `distance(obj_a, obj_b) <op> threshold` |
| `StateTrackerNodeType` | `char_state == target_state` (or NOT) |
| `ContactTrackerNodeType` | `obj in targets` contact check |
| `InputTrackerNodeType` | `inputs[action] == is_pressed` |
| `GameTimeTrackerNodeType` | `game_time <op> threshold` |
| `LogicAndNodeType` | `all(children)` |
| `LogicOrNodeType` | `any(children)` |
| `LogicNotNodeType` | `not child` |

### Key Files

| File | Purpose |
|------|---------|
| `Exp_Nodes/tracker_nodes.py` | Node UI definitions |
| `Exp_Game/interactions/exp_tracker_eval.py` | Serialization, world state, result application |
| `Exp_Game/engine/worker/entry.py` | Worker caches and evaluation logic |
| `Exp_Game/modal/exp_loop.py` | Job submission and result polling |

### Performance

- **Serialization**: Once at game start (no per-frame node traversal)
- **World state**: Minimal data (~100 bytes per object)
- **Evaluation**: Worker-side, parallel with main thread
- **Hz throttling**: Per-tracker, respects eval_hz setting (max 30)
- **Edge detection**: Only signal changes sent back to main thread

---

## IMPLEMENTATION PHASES

### Phase 1: Tracker Infrastructure (Engine-Side)

1. Define `TrackerDefinition` PropertyGroup
2. Add `scene.trackers` CollectionProperty
3. Implement tracker update logic in engine worker
4. Create snapshot pattern for object state
5. Test with CHARACTER_SPEED tracker

### Phase 2: Tracker Nodes (Design-Time)

1. Create `TrackerNode` base class
2. Implement CHARACTER_SPEED, CHARACTER_GROUNDED nodes
3. Implement OBJECT_PROPERTY tracker node
4. Implement DISTANCE tracker node
5. Add tracker output sockets

### Phase 3: Data Nodes

1. Create FloatNode (hardcoded + tracker binding)
2. Create IntNode
3. Create BoolNode
4. Create VectorNode
5. Create ObjectNode

### Phase 4: Dynamic Parameter Binding

1. Add `bindings` to InteractionDefinition
2. Add `bindings` to ReactionDefinition
3. Modify trigger nodes to have optional input sockets
4. Modify reaction nodes to have optional input sockets
5. Implement binding resolution in engine worker

### Phase 5: Conditions

1. Create ConditionNode (comparison logic)
2. Create AND, OR, NOT nodes
3. Create ConditionTrigger node
4. Implement condition evaluation in engine worker
5. Implement edge detection (ON_BECOME_TRUE)

### Phase 6: Advanced Trackers

1. CONTACT tracker (collision/touching state)
2. ANGLE tracker (angle between objects)
3. EXPRESSION tracker (computed from other trackers)
4. Custom property trackers

---

## KEY PRINCIPLES SUMMARY

1. **Node graph is design-time only** - Never traversed at runtime
2. **Engine worker does ALL computation** - Trackers, conditions, math
3. **Main thread only does bpy** - Reads and writes, nothing else
4. **Parameters can be hardcoded OR bound** - Flexible data flow
5. **Trackers are the runtime truth** - Updated every frame in worker
6. **Conditions enable complex logic** - Without runtime node traversal
7. **Performance is non-negotiable** - If it's not in the worker, it's wrong

---

## REMEMBER

The nodes are the **paintbrush**, not the **painting**.

They configure the system. They don't run the system.

The engine worker runs the system.
