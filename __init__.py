# __init__.py inside custom_nodes/tenosai_nodes/

# This file makes the 'tenosai_nodes' directory a Python package

# Import the node mappings from your main node file (update filename if you changed it)
try:
    # Assuming the main node file is now called llm_describe_node.py
    from .llm_describe_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print("\nTenosai LLM Image Describe node loaded successfully.\n")

except ImportError as e:
    print(f"\nError importing Tenosai LLM node: {e}\n")
    # Define empty mappings if import fails, so ComfyUI doesn't crash
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}


# Define the __all__ list to explicitly export the symbols ComfyUI needs
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# You could add other package-level logic here if needed in the future