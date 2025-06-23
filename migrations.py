#Exploratory/migrations.py

Your migration functions don’t need to “scan” your source files or import your MovementSettings class to know where to look—the Blender registration system has already attached your properties to the live scene object at runtime. Here’s the flow:

Registration happens first

You register MovementSettings as a PropertyGroup and attach it to the scene:

python
Copy
Edit
bpy.utils.register_class(MovementSettings)
bpy.types.Scene.game_schema = PointerProperty(type=GameSchema)
bpy.types.Scene.game_schema.movement = PointerProperty(type=MovementSettings)
Now every bpy.data.scenes[x].game_schema.movement is an instance of MovementSettings with whatever props (old and new) you’ve declared.

Migration runs on those live instances

Your run_migrations(scene) handler passes each scene into your migration functions.

Inside migrate_1_0_1_to_1_1_0(scene), you do:

python
Copy
Edit
ms = scene.game_schema.movement      # <— the live MovementSettings instance
if hasattr(ms, "run_speed"):         # <— checks if old prop is present
    ms.sprint_speed = ms.run_speed   # <— copies the stored data
    delattr(ms, "run_speed")         # <— removes the old attribute
You never touch your .py files here—this is purely data migration on the in-memory property objects.

No imports of your game-code needed

Because Blender has already created scene.game_schema.movement.run_speed at registration, your migration function can just use hasattr/getattr/delattr on that instance.

You only import CURRENT_VERSION (to compare versions) and the migration runner itself.



###########
okay i get it. what if i change an entire function? say a function that defines the way we look around, and i create a new function as well as a helper function that is attached to it. how would i make that migration work? or does it matter as long as it doesnt depend on old properties?


ChatGPT said:

Thought for 7 seconds
You only ever migrate the data that backs your functions, not the functions themselves. Any pure code change—rewriting how your camera‐look function works, adding helper functions, refactoring modules—just goes straight into your normal add-on code and ships with the new version. Old .blend files don’t “contain” your Python functions, they only contain the property values you stored, and when you reload the add-on, every scene will automatically pick up your new code.

When you do need a migration
You need to write a migration only if you change the schema—that is, any PropertyGroup or scene property that your functions rely on. For example:

Renaming a property

You rename run_speed → sprint_speed.

You write a migrate_1_0_1_to_1_1_0() that copies the old run_speed value into sprint_speed, then (optionally) deletes run_speed.

That’s purely data migration—nothing in your function code needs to know about it.

Adding a new property

You introduce crouch_speed.

If you give it a sensible default in its FloatProperty(...) definition, old scenes will automatically see that default. No migration needed.

If you want an old scene to start with a non-default value (say, half the walk speed), you can write a tiny migration that does ms.crouch_speed = ms.walk_speed * 0.5.

What if you restructure or rename functions?
Refactoring functions (moving code between modules, renaming helpers, changing signatures) doesn’t touch your scene data. Once the add-on reloads, every operator or timer that calls your function will just call the new version.

If you rename an operator ID (e.g. view3d.exp_modal → view3d.exploratory_modal), you can ship a one-release “alias” operator that simply redirects the old ID to the new one. Example:

python
Copy
Edit
class VIEW3D_OT_OldModalAlias(bpy.types.Operator):
    bl_idname = "view3d.exp_modal"           # old ID
    bl_label   = "Third Person Orbit (alias)"
    def execute(self, ctx):
        return bpy.ops.view3d.exploratory_modal('INVOKE_DEFAULT')
Then, in your next major version you drop the alias.

TL;DR
Code changes: just edit your functions/modules; old scenes pick up new code automatically.

Data changes (props renamed/added/removed): write tiny migration functions in your one migrations.py that move or initialize those values.

Your run_migrations(scene) loop only looks at scene.some_group.some_prop – it never needs to “know” where in your files the prop was declared.

As long as you don’t break the schema that your functions expect (or you migrate it), your logic will run un-broken—no hunting through every file required