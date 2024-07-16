from PIL import Image, ImageDraw, ImageFont
import math


def visualize_tree(
    node, filename="tree_visualization.png", node_radius=40, level_height=150
):
    def get_tree_width(node):
        if not hasattr(node, "childs") or not node.childs:
            return 1
        return sum(
            get_tree_width(child) if hasattr(child, "childs") else 1
            for child in node.childs.values()
        )

    def draw_node(draw, x, y, node, direction=""):
        nonlocal max_x, max_y
        max_x = max(max_x, x)
        max_y = max(max_y, y)

        # Draw the node
        draw.ellipse(
            [
                x - node_radius,
                y - node_radius,
                x + node_radius,
                y + node_radius,
            ],
            outline="black",
            fill="white",
        )

        font = ImageFont.load_default()
        text = f"{direction}\n{getattr(node, 'dist_min_to_target', 'N/A')}"
        draw.multiline_textbbox(
            (x, y), text, font=font, anchor="mm"
        )
        draw.multiline_text(
            (x, y), text, fill="black", font=font, anchor="mm", align="center"
        )

        if hasattr(node, "childs") and node.childs:
            child_width = get_tree_width(node) * node_radius * 2
            start_x = x - child_width / 2 + node_radius

            for direction, child in node.childs.items():
                if hasattr(child, "childs"):
                    child_x = start_x + get_tree_width(child) * node_radius
                    child_y = y + level_height
                    draw.line(
                        [x, y + node_radius, child_x, child_y - node_radius],
                        fill="black",
                    )
                    draw_node(draw, child_x, child_y, child, direction)
                    start_x += get_tree_width(child) * node_radius * 2
                else:
                    leaf_x = start_x + node_radius
                    leaf_y = y + level_height
                    draw.line(
                        [x, y + node_radius, leaf_x, leaf_y - node_radius],
                        fill="black",
                    )
                    draw.ellipse(
                        [
                            leaf_x - node_radius,
                            leaf_y - node_radius,
                            leaf_x + node_radius,
                            leaf_y + node_radius,
                        ],
                        outline="black",
                        fill="lightgray",
                    )
                    draw.text(
                        (leaf_x, leaf_y),
                        f"{direction}\n{child}",
                        fill="black",
                        font=font,
                        anchor="mm",
                        align="center",
                    )
                    start_x += node_radius * 2
                    max_x = max(max_x, leaf_x)
                    max_y = max(max_y, leaf_y)

    max_x, max_y = 0, 0
    img = Image.new("RGB", (1, 1), color="white")
    draw = ImageDraw.Draw(img)

    draw_node(draw, 0, node_radius, node)

    img_width = max_x + node_radius * 20
    img_height = max_y + node_radius * 10
    img = Image.new(
        "RGB", (math.ceil(img_width), math.ceil(img_height)), color="white"
    )
    draw = ImageDraw.Draw(img)

    draw_node(draw, img_width / 2, node_radius, node)

    # img.save(filename)
    img.show()


# Example usage:
# visualize_tree(obs[0])
