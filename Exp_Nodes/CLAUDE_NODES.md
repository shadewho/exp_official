# CLAUDE_NODES.md

Guide for working with the Exploratory node system.

**REQUIRED:** Read `../Exp_Game/CLAUDE.md` for ENGINE-FIRST architecture.

---

## 🎯 THE GOAL: Procedural & Dynamic Node System

Build a **universal, procedural system** where:
- All data types have **one socket type** that connects to any compatible socket
- Data flows freely between nodes (Float → Float, Vector → Vector, Bool → Bool)
- The system is **scalable** - add new nodes that plug into existing infrastructure
- Everything is **design-time configuration** - nodes are never read during gameplay

**Universal Socket Types (Implemented):**
```
ExpBoolSocketType   → ALL bool connections
ExpFloatSocketType  → ALL float connections
ExpVectorSocketType → ALL vector connections (location, rotation, scale, any 3D)
ExpObjectSocketType → ALL object references
```

Any node outputting a Float can connect to ANY node accepting a Float. Same for all types.

---

## 🚫 CRITICAL: NEVER Read Nodes at Runtime

**The node graph is ONLY for design-time configuration. It is NEVER traversed during gameplay.**

```
❌ NEVER DO THIS:
def execute_reaction(reaction):
    # Find the node, traverse connections, read values
    for node in node_tree.nodes:
        if node.is_linked_to(reaction):
            value = node.get_connected_value()  # WRONG!

✅ ALWAYS DO THIS:
def execute_reaction(reaction):
    # Use pre-serialized bindings (baked at game start)
    value = resolve_vector(reaction, "location", reaction.location)
```

**Why this matters:**
- Node traversal is SLOW (O(n) connections, Python overhead)
- Blender node API has overhead on every access
- Main thread must stay lean for 30Hz gameplay
- Workers can't access bpy anyway

**The Pattern:**
```
DESIGN TIME              GAME START                 RUNTIME
┌─────────────┐         ┌─────────────────┐        ┌─────────────────┐
│ Node Graph  │ ──────► │ Serialize to    │ ──────►│ Fast dict       │
│ (visual UI) │         │ flat data       │        │ lookups only    │
└─────────────┘         └─────────────────┘        └─────────────────┘
     USER                   ONCE                      EVERY FRAME
```

---

## 🔗 Implemented Systems

### 1. Tracker System (Worker-Offloaded)

Trackers evaluate conditions in the engine worker, not main thread.

**Flow:**
1. **Game Start:** `serialize_tracker_graph()` → flat condition trees
2. **Each Frame:** `collect_world_state()` → positions, inputs, char_state
3. **Worker:** `_eval_condition_tree()` → evaluate conditions
4. **Result:** `apply_tracker_results()` → set `external_signal` on triggers

**Supported Trackers:**
- `DistanceTrackerNodeType` - distance between objects
- `StateTrackerNodeType` - character state (grounded, jumping, etc.)
- `ContactTrackerNodeType` - collision contacts
- `InputTrackerNodeType` - key presses
- `GameTimeTrackerNodeType` - elapsed time
- `LogicAnd/Or/NotNodeType` - combine conditions

**Key Files:**
- `Exp_Game/interactions/exp_tracker_eval.py` - serialization & results
- `Exp_Game/engine/worker/interactions/trackers.py` - worker evaluation

### 2. Reaction Binding System (Implemented)

Data nodes can connect to reaction inputs (Transform location, duration, etc.)

**Flow:**
1. **Game Start:** `serialize_reaction_bindings()` → scan connections, bake values
2. **Runtime:** `resolve_vector(reaction, "param", default)` → fast lookup

**Supported:**
```python
resolve_vector(reaction, "transform_location", reaction.transform_location)
resolve_euler(reaction, "transform_rotation", reaction.transform_rotation)
resolve_float(reaction, "transform_duration", reaction.transform_duration)
resolve_object(reaction, "target", reaction.target_object)
resolve_bool(reaction, "enabled", reaction.enabled)
resolve_int(reaction, "count", reaction.count)
```

**Key Files:**
- `Exp_Game/reactions/exp_bindings.py` - serialization & resolution
- `Exp_Game/reactions/exp_transforms.py` - uses bindings

### 3. Data Nodes (Implemented)

Universal value sources that connect to any matching socket:

| Node | Socket Type | Property |
|------|-------------|----------|
| `FloatDataNode` | `ExpFloatSocketType` | `value` |
| `IntDataNode` | `ExpIntSocketType` | `value` |
| `BoolDataNode` | `ExpBoolSocketType` | `value` |
| `FloatVectorDataNode` | `ExpVectorSocketType` | `value` |
| `ObjectDataNode` | `ExpObjectSocketType` | `target_object` / `use_character` |

---

## 🚨 State Reset Requirements

**ALL state MUST be reset on game start/reset/end:**

| State | Location | Reset Function |
|-------|----------|----------------|
| `_tracker_primed` | worker | `handle_cache_trackers()` |
| `_tracker_states` | worker | `handle_cache_trackers()` |
| `_tracker_generation` | main | `reset_worker_trackers()` |
| `_reaction_bindings` | main | `reset_bindings()` |
| `inter.external_signal` | scene | `reset_all_interactions()` |
| `inter.has_fired` | scene | `reset_all_interactions()` |

**Why:** Stale state causes false triggers. The `external_signal` bug (fixed today) caused triggers to fire immediately after reset because the signal wasn't cleared.

---

## ⚠️ Node Layout Architecture (CRITICAL)

**All nodes MUST follow the inline socket pattern. Sockets are drawn WITH their data fields, not separately.**

### The Golden Rule

```
✅ CORRECT - Inline Layout:          ❌ WRONG - Separated Layout:
┌──────────────────────────┐        ┌──────────────────────────┐
│  Distance Tracker        │        │  Distance Tracker        │
├──────────────────────────┤        ├──────────────────────────┤
│ ● Object A: [Cube    ▼]  │        │  Object A: [Cube    ▼]   │
│ ● Object B: [Sphere  ▼]  │        │  Object B: [Sphere  ▼]   │
│   Distance < [5.0    ]   │        │  Distance < [5.0    ]    │
│   Hz: [10]               │        │  Hz: [10]                │
├──────────────────────────┤        │                          │
│              Condition ○ │        │  ● Object A              │
└──────────────────────────┘        │  ● Object B              │
                                    │              Condition ○ │
                                    └──────────────────────────┘
       Socket ● is NEXT to               Sockets are BELOW
       its property field                their property fields
```

### Why This Matters

1. **Visual clarity** - User sees socket and its field together
2. **Connection feedback** - When connected, the field hides (socket takes over)
3. **Compact nodes** - No wasted vertical space
4. **Intuitive wiring** - Wire goes TO the field it affects

### Implementation Pattern

**Socket classes define `draw()` to render inline with properties:**

```python
class ExpVectorSocketType(NodeSocket):
    bl_idname = "ExpVectorSocketType"
    bl_label = "Vector"

    # Property shown when NOT connected
    value: FloatVectorProperty(subtype='XYZ')

    # Links to reaction property for binding system
    reaction_prop: StringProperty()

    def draw(self, context, layout, node, text):
        if self.is_linked:
            # Connected - just show socket name
            layout.label(text=text)
        elif self.reaction_prop and hasattr(node, 'reaction'):
            # Reaction node - show the reaction's property inline
            layout.prop(node.reaction, self.reaction_prop, text=text)
        else:
            # Data node - show socket's own value
            layout.prop(self, "value", text=text)

    def draw_color(self, context, node):
        return (0.4, 0.4, 0.8, 1.0)  # Blue for vectors
```

**Node classes use socket's `prop_name` or `reaction_prop`:**

```python
class DistanceTrackerNodeType(Node):
    bl_idname = "DistanceTrackerNodeType"
    bl_label = "Distance Tracker"

    # Non-socket properties (no connection needed)
    eval_hz: IntProperty(name="Hz", default=10, min=1, max=60)

    def init(self, context):
        # Input sockets - will draw inline with their fields
        obj_a = self.inputs.new("ExpObjectSocketType", "Object A")
        obj_a.prop_name = "object_a"  # Property on THIS node

        obj_b = self.inputs.new("ExpObjectSocketType", "Object B")
        obj_b.prop_name = "object_b"

        dist = self.inputs.new("ExpFloatSocketType", "Distance")
        dist.prop_name = "distance_threshold"

        # Output socket
        self.outputs.new("ExpBoolSocketType", "Condition")

    def draw_buttons(self, context, layout):
        # Only draw properties WITHOUT sockets
        layout.prop(self, "eval_hz")
        # DO NOT draw object_a, object_b, distance here!
        # The sockets handle those in their draw() method
```

### Reaction Node Pattern

**Reaction nodes link sockets to reaction properties via `reaction_prop`:**

```python
class TransformReactionNodeType(Node):
    bl_idname = "TransformReactionNodeType"
    bl_label = "Transform"

    @property
    def reaction(self):
        # Get the actual reaction from scene.reactions
        return get_reaction_by_index(self.reaction_index)

    def init(self, context):
        # Each socket's reaction_prop points to the reaction's property
        loc = self.inputs.new("ExpVectorSocketType", "Location")
        loc.reaction_prop = "transform_location"

        rot = self.inputs.new("ExpVectorSocketType", "Rotation")
        rot.reaction_prop = "transform_rotation"

        dur = self.inputs.new("ExpFloatSocketType", "Duration")
        dur.reaction_prop = "transform_duration"

    def draw_buttons(self, context, layout):
        r = self.reaction
        if not r:
            layout.label(text="No reaction linked")
            return

        # Only draw non-socket properties
        layout.prop(r, "transform_mode")
        layout.prop(r, "easing")
        # Location, Rotation, Duration are drawn by their sockets!
```

### Socket Type Reference

| Socket Type | Color | Use For |
|-------------|-------|---------|
| `ExpBoolSocketType` | Red | Conditions, toggles, enables |
| `ExpFloatSocketType` | Gray | Numbers, durations, distances |
| `ExpIntSocketType` | Green | Counts, indices |
| `ExpVectorSocketType` | Blue | Positions, rotations, scales |
| `ExpObjectSocketType` | Orange | Blender object references |

### Data Node Pattern (Value Sources)

```python
class FloatVectorDataNode(Node):
    bl_idname = "FloatVectorDataNode"
    bl_label = "Vector"

    value: FloatVectorProperty(name="Value", subtype='XYZ')

    def init(self, context):
        out = self.outputs.new("ExpVectorSocketType", "Vector")
        out.prop_name = "value"  # Socket draws this node's value property

    def draw_buttons(self, context, layout):
        # Nothing here - socket handles the value display
        pass
```

### Checklist for New Nodes

- [ ] All connectable properties have a socket
- [ ] Socket's `prop_name` or `reaction_prop` is set correctly
- [ ] `draw_buttons()` only draws non-socket properties
- [ ] Socket `draw()` method shows property when not connected
- [ ] Socket `draw()` method shows label only when connected

---

## 📁 File Structure

```
Exp_Nodes/
├── utility_nodes.py      ← Data nodes (Float, Vector, Object, etc.)
├── tracker_nodes.py      ← Tracker nodes (Distance, State, Input, etc.)
├── trigger_nodes.py      ← Trigger nodes (External, Proximity, etc.)
├── reaction_nodes.py     ← Reaction nodes (Transform, Sound, etc.)
└── node_editor.py        ← Tree, unified sockets, menus

Exp_Game/
├── reactions/exp_bindings.py           ← Binding serialization & resolution
├── interactions/exp_tracker_eval.py    ← Tracker serialization & results
├── engine/worker/interactions/trackers.py ← Worker-side evaluation
└── startup_and_reset/exp_game_reset.py ← State reset
```

---

## 🔧 Adding Bindings to New Reactions

```python
# 1. In reaction node init(), create socket with reaction_prop:
s = self.inputs.new("ExpVectorSocketType", "Location")
s.reaction_prop = "my_location_prop"

# 2. In executor, use resolve function:
from ..reactions.exp_bindings import resolve_vector

def execute_my_reaction(reaction):
    loc = resolve_vector(reaction, "my_location_prop", reaction.my_location_prop)
```

---

## 📋 Today's Fixes (Session Summary)

1. **Tracker Priming** - Added `_tracker_primed` set to prevent false trigger on first evaluation
2. **Generation Counter** - Added `_tracker_generation` to discard stale results after reset
3. **External Signal Reset** - Fixed `inter.external_signal` not being cleared on reset
4. **Reaction Bindings** - Implemented full binding system for Transform reactions
5. **Universal Sockets** - Unified all socket types so same-type nodes can connect

---

## 💡 Core Principle

**Nodes are the paintbrush, not the painting.**

They configure the system at design-time. The engine worker runs the system at runtime.

Never traverse nodes during gameplay. Always serialize to flat data structures.
