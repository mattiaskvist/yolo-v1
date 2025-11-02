from manim import (
    DOWN,
    GRAY,
    RED,
    RIGHT,
    UL,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Create,
    Dot,
    FadeIn,
    Flash,
    Rectangle,
    Scene,
    Square,
    Text,
    VGroup,
    Write,
    GRAY_B,
    LEFT,
    Line,
    Brace,
    BLUE,
    GRAY_C,
    ORANGE,
    GREEN,
    TransformFromCopy,
    FadeOut,
    MathTex,
    Indicate,
    SurroundingRectangle,
    PURPLE,
    PINK,
    ReplacementTransform,
    DashedLine,
    Wiggle,
)


class YoloGridAndResponsibility(Scene):
    """
    Animates the S x S grid and highlights the "responsible" cell
    based on the object's center.
    """

    def construct(self):
        S = 7  # The 7x7 grid
        IMG_SIDE_LENGTH = 5.5

        # --- 1. Title ---
        title = Text("YOLOv1: The S x S Grid").scale(0.8).to_edge(UP, buff=0.3)
        self.play(Write(title))

        # --- 2. Create the 'Image' ---
        image_rect = Square(
            side_length=IMG_SIDE_LENGTH, color=WHITE, stroke_width=2
        ).shift(LEFT * 2.5)

        image_label = Text("Input Image").scale(0.6).next_to(image_rect, DOWN, buff=0.3)
        self.play(Create(image_rect), FadeIn(image_label))
        self.wait(0.5)

        # --- 3. Create the Grid ---
        grid = VGroup(
            *[
                Square(
                    side_length=IMG_SIDE_LENGTH / S, stroke_width=1, stroke_color=GRAY
                )
                for _ in range(S * S)
            ]
        ).arrange_in_grid(S, S, buff=0)
        grid.move_to(image_rect)

        grid_text = Text(f"{S}x{S} Grid").scale(0.7).next_to(image_label)
        self.play(Create(grid), image_label.animate.next_to(image_rect, DOWN, buff=0.8))
        self.wait(1)

        # --- 4. Add a Ground Truth Object (a "dog") ---
        # (x, y, w, h) in relative image coordinates (0.0 to 1.0)
        gt_center_rel = (0.62, 0.75)  # 62% across, 75% down
        gt_dims_rel = (0.3, 0.4)  # 30% width, 40% height

        # Convert relative coords to Manim's absolute coords
        img_origin_ul = image_rect.get_corner(UL)  # Top-Left corner

        gt_center_abs = (
            img_origin_ul
            + RIGHT * gt_center_rel[0] * IMG_SIDE_LENGTH
            + DOWN * gt_center_rel[1] * IMG_SIDE_LENGTH
        )

        gt_box = Rectangle(
            width=gt_dims_rel[0] * IMG_SIDE_LENGTH,
            height=gt_dims_rel[1] * IMG_SIDE_LENGTH,
            color=RED,
            stroke_width=3,
        ).move_to(gt_center_abs)  # .move_to() uses the center

        gt_center_dot = Dot(gt_center_abs, color=RED, radius=0.08)
        gt_label = Text("Dog").scale(0.6).next_to(gt_box, UP, buff=0.1)

        self.play(Create(gt_box), FadeIn(gt_center_dot), Write(gt_label))
        self.wait(1)

        # --- 5. Find and Highlight the Responsible Cell ---
        # This is the "Aha!" moment from your question.

        # Calculate which (i, j) cell this center is in
        cell_i = int(gt_center_rel[1] * S)  # Row (y)
        cell_j = int(gt_center_rel[0] * S)  # Column (x)

        # Get the cell from our VGroup
        responsible_cell_index = cell_i * S + cell_j
        responsible_cell = grid[responsible_cell_index]

        highlight_cell = responsible_cell.copy().set(
            fill_color=YELLOW, fill_opacity=0.5, stroke_width=0
        )

        resp_text = Text(
            "Cell (i, j) is responsible\nfor this object", t2c={"(i, j)": YELLOW}
        )
        resp_text.scale(0.6).shift(RIGHT * 3.5)

        arrow = Arrow(
            resp_text.get_left(), highlight_cell.get_center(), buff=0.2, color=YELLOW
        )

        self.play(
            Flash(gt_center_dot, color=YELLOW, flash_radius=0.5, line_length=0.3),
            FadeIn(highlight_cell),
        )
        self.play(Write(resp_text), Create(arrow))
        self.wait(3)


class YoloCoordinates(Scene):
    """
    Animates how (x, y, w, h) are predicted.
    - (x, y) are relative to the cell
    - (w, h) are relative to the image
    """

    def construct(self):
        S = 7
        IMG_WIDTH = 4.0

        # --- 1. Setup Context ---
        title = Text("YOLOv1: Coordinate Prediction").scale(0.8).to_edge(UP, buff=0.3)
        self.play(Write(title))

        # A. Show the full image/grid (small, for context)
        image_rect = Rectangle(width=IMG_WIDTH, height=IMG_WIDTH, color=GRAY_B).shift(
            LEFT * 4.5 + DOWN * 0.5
        )

        grid = (
            VGroup(
                *[
                    Square(
                        side_length=IMG_WIDTH / S, stroke_width=1, stroke_color=GRAY_C
                    )
                    for _ in range(S * S)
                ]
            )
            .arrange_in_grid(S, S, buff=0)
            .move_to(image_rect)
        )

        image_label = (
            Text("Full Image Context", color=GRAY)
            .scale(0.5)
            .next_to(image_rect, DOWN, buff=0.2)
        )
        self.add(image_rect, grid, image_label)

        # B. Show a "zoomed-in" cell
        zoomed_cell = Square(side_length=4.0, color=WHITE, stroke_width=3).shift(
            RIGHT * 2.5 + UP * 0.5
        )
        cell_label = (
            Text("Zoomed-in Cell (i, j)").scale(0.7).next_to(zoomed_cell, UP, buff=0.2)
        )
        self.play(Create(zoomed_cell), Write(cell_label))

        # --- 2. Define Predicted (x, y) ---
        # These are the *model outputs*, between 0 and 1
        x_pred_val = 0.7
        y_pred_val = 0.4

        cell_origin_ul = zoomed_cell.get_corner(UL)

        obj_center_abs = (
            cell_origin_ul
            + RIGHT * x_pred_val * zoomed_cell.width
            + DOWN * y_pred_val * zoomed_cell.height
        )

        obj_dot = Dot(obj_center_abs, color=RED, radius=0.08)

        # --- 3. Animate (x, y) relative to CELL ---
        x_line = Line(
            cell_origin_ul,
            cell_origin_ul + RIGHT * x_pred_val * zoomed_cell.width,
            color=BLUE,
            stroke_width=5,
        )
        x_brace = Brace(x_line, direction=DOWN, buff=0.2)
        x_label = Text(f"x = {x_pred_val}").scale(0.6).next_to(x_brace, DOWN)

        y_line = Line(x_line.get_end(), obj_dot, color=ORANGE, stroke_width=5)
        y_brace = Brace(y_line, direction=RIGHT, buff=0.2)
        y_label = Text(f"y = {y_pred_val}").scale(0.6).next_to(y_brace, RIGHT)

        xy_text = Text(
            "(x, y) are offsets\nrelative to the cell", t2c={"(x, y)": YELLOW}
        )
        xy_text.scale(0.6).shift(DOWN * 3.2 + RIGHT * 2.5)

        self.play(FadeIn(obj_dot), Write(xy_text))
        self.play(Create(x_line), FadeIn(x_brace), FadeIn(x_label), run_time=1.5)
        self.play(Create(y_line), FadeIn(y_brace), FadeIn(y_label), run_time=1.5)
        self.wait(1)

        # --- 4. Define Predicted (w, h) ---
        w_pred_val = 0.3  # 30% of IMAGE width
        h_pred_val = 0.4  # 40% of IMAGE height

        # --- 5. Animate (w, h) relative to IMAGE ---

        # A. Show the predicted box
        pred_box = Rectangle(
            width=w_pred_val * IMG_WIDTH,  # <-- KEY! Based on IMG_WIDTH
            height=h_pred_val * IMG_WIDTH,  # <-- KEY! Based on IMG_WIDTH
            color=GREEN,
            stroke_width=3,
        ).move_to(obj_dot)  # Center it on the (x,y) point

        self.play(Create(pred_box))

        # B. Show braces on the IMAGE to represent w=1.0
        w_brace_img = Brace(image_rect, direction=DOWN, buff=0.1, color=GRAY)
        w_brace_img_label = (
            Text("Image Width = 1.0", color=GRAY).scale(0.5).next_to(w_brace_img, DOWN)
        )

        # C. Show braces on the BOX relative to the image
        w_brace_box = Brace(pred_box, direction=DOWN, buff=0.1, color=GREEN)
        w_brace_box_label = (
            Text(f"w = {w_pred_val}").scale(0.6).next_to(w_brace_box, DOWN)
        )

        h_brace_box = Brace(pred_box, direction=LEFT, buff=0.1, color=GREEN)
        h_brace_box_label = (
            Text(f"h = {h_pred_val}").scale(0.6).next_to(h_brace_box, LEFT)
        )

        wh_text = Text(
            "(w, h) are percentages\nof the TOTAL IMAGE", t2c={"(w, h)": YELLOW}
        )
        wh_text.scale(0.6).shift(DOWN * 3.2 + LEFT * 2.5)

        self.play(FadeIn(wh_text))
        self.play(
            FadeIn(w_brace_img),
            FadeIn(w_brace_img_label),
        )
        self.play(
            TransformFromCopy(w_brace_img, w_brace_box),
            TransformFromCopy(w_brace_img_label, w_brace_box_label),
        )
        self.play(FadeIn(h_brace_box), FadeIn(h_brace_box_label))

        self.wait(3)


class PredictionTensor(Scene):
    """
    Scene 2: Shows what each responsible cell predicts.
    Demonstrates the prediction tensor splitting into box predictions and class probabilities.
    """

    def construct(self):
        B = 2  # Number of bounding boxes
        C = 20  # Number of classes
        TENSOR_LENGTH = B * 5 + C  # 30

        # --- 1. Title ---
        title = Text("YOLOv1: The Prediction Tensor").scale(0.8).to_edge(UP, buff=0.3)
        self.play(Write(title))

        # --- 2. Show a highlighted cell ---
        cell = Square(side_length=2.0, color=YELLOW, fill_opacity=0.3).shift(
            LEFT * 4 + UP * 0.5
        )
        cell_label = Text("Responsible Cell").scale(0.6).next_to(cell, UP, buff=0.2)
        self.play(Create(cell), Write(cell_label))
        self.wait(0.5)

        # --- 3. Pull out the prediction vector ---
        vector_start = cell.get_right()
        vector_end = vector_start + RIGHT * 6

        # Create a long rectangle to represent the tensor
        tensor_rect = Rectangle(
            width=6.0, height=0.4, color=WHITE, fill_opacity=0.2
        ).move_to((vector_start + vector_end) / 2 + UP * 0.5)

        arrow = Arrow(vector_start, tensor_rect.get_left(), buff=0.1, color=WHITE)
        tensor_label = Text(f"Prediction Tensor (length {TENSOR_LENGTH})").scale(0.5)
        tensor_label.next_to(tensor_rect, UP, buff=0.2)

        self.play(Create(arrow), Create(tensor_rect), Write(tensor_label))
        self.wait(1)

        # --- 4. Split into three chunks ---
        # Calculate widths proportionally
        box1_width = (5 / TENSOR_LENGTH) * 6.0
        box2_width = (5 / TENSOR_LENGTH) * 6.0
        class_width = (C / TENSOR_LENGTH) * 6.0

        # Create the three chunks
        chunk1 = Rectangle(
            width=box1_width, height=0.4, color=BLUE, fill_opacity=0.5
        ).next_to(tensor_rect.get_left(), RIGHT, buff=0)
        chunk1.align_to(tensor_rect, DOWN)

        chunk2 = Rectangle(
            width=box2_width, height=0.4, color=GREEN, fill_opacity=0.5
        ).next_to(chunk1, RIGHT, buff=0)
        chunk2.align_to(tensor_rect, DOWN)

        chunk3 = Rectangle(
            width=class_width, height=0.4, color=RED, fill_opacity=0.5
        ).next_to(chunk2, RIGHT, buff=0)
        chunk3.align_to(tensor_rect, DOWN)

        # Labels for chunks
        chunk1_label = Text("Box 1\n(x,y,w,h,c)", color=BLUE).scale(0.4)
        chunk1_label.next_to(chunk1, DOWN, buff=0.4)

        chunk2_label = Text("Box 2\n(x,y,w,h,c)", color=GREEN).scale(0.4)
        chunk2_label.next_to(chunk2, DOWN, buff=0.4)

        chunk3_label = Text("Class Probs\n(20 values)", color=RED).scale(0.4)
        chunk3_label.next_to(chunk3, DOWN, buff=0.4)

        self.play(FadeOut(tensor_rect), Create(chunk1), Create(chunk2), Create(chunk3))
        self.play(Write(chunk1_label), Write(chunk2_label), Write(chunk3_label))
        self.wait(1)

        # --- 5. Add explanation ---
        explanation = Text(
            f"Each cell predicts {B} boxes and one set of class probabilities",
            t2c={f"{B}": YELLOW, "class probabilities": RED},
        ).scale(0.6)
        explanation.to_edge(DOWN)
        self.play(Write(explanation))
        self.wait(3)


class ConfidenceScoreFiltering(Scene):
    """
    Scene 4: Shows how to compute final confidence and filter boxes.
    """

    def construct(self):
        # --- 1. Title ---
        title = (
            Text("YOLOv1: Confidence Score & Filtering")
            .scale(0.8)
            .to_edge(UP, buff=0.3)
        )
        self.play(Write(title))

        # --- 2. Show the tensor chunks (recap from Scene 2) ---
        chunk1 = Rectangle(width=1.5, height=0.6, color=BLUE, fill_opacity=0.3)
        chunk1.shift(LEFT * 3.5 + UP * 2)
        chunk1_label = (
            Text("Box 1", color=BLUE).scale(0.5).next_to(chunk1, UP, buff=0.2)
        )

        chunk3 = Rectangle(width=2.5, height=0.6, color=RED, fill_opacity=0.3)
        chunk3.shift(RIGHT * 2.5 + UP * 2)
        chunk3_label = Text("Class Probabilities", color=RED).scale(0.5)
        chunk3_label.next_to(chunk3, UP, buff=0.2)

        self.play(
            Create(chunk1), Write(chunk1_label), Create(chunk3), Write(chunk3_label)
        )
        self.wait(0.5)

        # --- 3. Highlight objectness score ---
        objectness_text = Text("c = 0.9", color=BLUE).scale(0.7)
        objectness_text.move_to(chunk1)
        self.play(Write(objectness_text))
        self.wait(0.5)

        # --- 4. Highlight class probability ---
        class_prob_text = Text("P(dog) = 0.95", color=RED).scale(0.7)
        class_prob_text.move_to(chunk3)
        self.play(Write(class_prob_text))
        self.wait(0.5)

        # --- 5. Show the formula ---
        formula = Text("final_conf = objectness × class_prob").scale(0.6)
        formula.shift(DOWN * 0.2)
        self.play(Write(formula))
        self.wait(1)

        # --- 6. Calculate result ---
        calculation = Text("= 0.9 × 0.95 = 0.855").scale(0.6)
        calculation.next_to(formula, DOWN, buff=0.3)
        self.play(Write(calculation))
        self.wait(1)

        # --- 7. Show filtering with threshold ---
        # Clear previous elements first
        self.play(FadeOut(VGroup(formula, calculation)))

        threshold_line = DashedLine(LEFT * 5, RIGHT * 5, color=YELLOW)
        threshold_line.shift(DOWN * 0.5)
        threshold_label = Text("Threshold = 0.5", color=YELLOW).scale(0.5)
        threshold_label.next_to(threshold_line, LEFT, buff=0.3)

        self.play(Create(threshold_line), Write(threshold_label))

        # Show example boxes with scores
        boxes_group = VGroup()
        for i, (score, y_offset) in enumerate(
            [(0.855, -0.8), (0.72, -1.2), (0.45, -1.6), (0.23, -2.0)]
        ):
            color = GREEN if score >= 0.5 else GRAY
            box_rect = Rectangle(
                width=1.0, height=0.4, color=color, fill_opacity=0.3
            ).shift(RIGHT * 0 + DOWN * y_offset)
            score_text = Text(f"{score}", color=color).scale(0.45)
            score_text.move_to(box_rect)
            boxes_group.add(box_rect, score_text)

        self.play(FadeIn(boxes_group))

        explanation = (
            Text(
                "Keep boxes above threshold, discard the rest",
                t2c={"above": GREEN, "discard": GRAY},
            )
            .scale(0.55)
            .to_edge(DOWN, buff=0.3)
        )
        self.play(Write(explanation))
        self.wait(3)


class NonMaximalSuppression(Scene):
    """
    Scene 5: Demonstrates NMS to clean up overlapping boxes.
    """

    def construct(self):
        # --- 1. Title ---
        title = (
            Text("YOLOv1: Non-Maximum Suppression (NMS)")
            .scale(0.75)
            .to_edge(UP, buff=0.3)
        )
        self.play(Write(title))

        # --- 2. Show an image area with overlapping boxes ---
        image_area = Rectangle(width=5.5, height=3.5, color=GRAY_B).shift(
            LEFT * 2.5 + DOWN * 0.3
        )
        self.play(Create(image_area))

        # Three overlapping boxes for "dog"
        box_a = Rectangle(width=2.0, height=1.5, color=GREEN, stroke_width=4)
        box_a.move_to(image_area.get_center() + UP * 0.2)
        label_a = Text("A: 0.85", color=GREEN).scale(0.5).next_to(box_a, UP, buff=0.1)

        box_b = Rectangle(width=2.1, height=1.6, color=BLUE, stroke_width=4)
        box_b.move_to(image_area.get_center() + RIGHT * 0.3)
        label_b = Text("B: 0.75", color=BLUE).scale(0.5).next_to(box_b, UP, buff=0.1)

        box_c = Rectangle(width=1.8, height=1.4, color=ORANGE, stroke_width=4)
        box_c.move_to(image_area.get_center() + LEFT * 0.2 + DOWN * 0.1)
        label_c = Text("C: 0.60", color=ORANGE).scale(0.5).next_to(box_c, UP, buff=0.1)

        self.play(
            Create(box_a),
            Write(label_a),
            Create(box_b),
            Write(label_b),
            Create(box_c),
            Write(label_c),
        )
        self.wait(1)

        # --- 3. Show NMS process on the right side ---
        process_title = Text("NMS Process:").scale(0.55).shift(RIGHT * 4 + UP * 2.5)
        self.play(Write(process_title))

        # Step 1: Sort by confidence
        step1 = Text("1. Keep box A (highest)", color=GREEN).scale(0.45)
        step1.next_to(process_title, DOWN, aligned_edge=LEFT, buff=0.25)
        self.play(Write(step1), Flash(box_a, color=GREEN))
        self.wait(1)

        # Step 2: Check IoU with B
        step2 = Text("2. IoU(A,B) = 0.9 > 0.5", color=BLUE).scale(0.45)
        step2.next_to(step1, DOWN, aligned_edge=LEFT, buff=0.25)
        self.play(Write(step2))
        self.wait(0.5)

        step2b = Text("   → Suppress B", color=GRAY).scale(0.45)
        step2b.next_to(step2, DOWN, aligned_edge=LEFT, buff=0.15)
        self.play(
            Write(step2b),
            box_b.animate.set_stroke(color=GRAY, opacity=0.3),
            label_b.animate.set_color(GRAY).set_opacity(0.3),
        )
        self.wait(1)

        # Step 3: Check IoU with C
        step3 = Text("3. IoU(A,C) = 0.8 > 0.5", color=ORANGE).scale(0.45)
        step3.next_to(step2b, DOWN, aligned_edge=LEFT, buff=0.25)
        self.play(Write(step3))
        self.wait(0.5)

        step3b = Text("   → Suppress C", color=GRAY).scale(0.45)
        step3b.next_to(step3, DOWN, aligned_edge=LEFT, buff=0.15)
        self.play(
            Write(step3b),
            box_c.animate.set_stroke(color=GRAY, opacity=0.3),
            label_c.animate.set_color(GRAY).set_opacity(0.3),
        )
        self.wait(1)

        # Final result
        result = Text("Final: Keep only Box A!", color=GREEN).scale(0.5)
        result.next_to(step3b, DOWN, buff=0.4)
        self.play(Write(result), Flash(box_a, color=GREEN, flash_radius=1.2))
        self.wait(3)


class FinalResult(Scene):
    """
    Scene 6: Shows the clean final output.
    """

    def construct(self):
        # --- 1. Show messy intermediate state ---
        title_messy = (
            Text("Before NMS: Messy Predictions").scale(0.75).to_edge(UP, buff=0.3)
        )
        self.play(Write(title_messy))

        # Create image area
        image = Rectangle(width=10, height=5, color=WHITE, stroke_width=2).shift(
            DOWN * 0.3
        )
        self.play(Create(image))

        # Show many overlapping boxes
        messy_boxes = VGroup()
        import random

        random.seed(42)  # For consistent positioning
        for _ in range(8):
            x = random.uniform(-4, 4)
            y = random.uniform(-1.8, 1.5)
            w = random.uniform(1, 2)
            h = random.uniform(0.8, 1.5)
            box = Rectangle(
                width=w, height=h, color=YELLOW, stroke_width=2, fill_opacity=0.1
            ).shift(RIGHT * x + UP * y + DOWN * 0.3)
            messy_boxes.add(box)

        self.play(Create(messy_boxes))
        self.wait(1)

        # --- 2. Transition to clean output ---
        self.play(FadeOut(messy_boxes), FadeOut(title_messy))

        title_clean = (
            Text("After NMS: Clean Detections").scale(0.75).to_edge(UP, buff=0.3)
        )
        self.play(Write(title_clean))

        # Show final clean boxes
        dog_box = Rectangle(width=2.5, height=2.0, color=GREEN, stroke_width=4)
        dog_box.shift(LEFT * 2.5 + DOWN * 0.3)
        dog_label = Text("Dog (0.85)", color=GREEN).scale(0.6)
        dog_label.next_to(dog_box, UP, buff=0.15)

        bicycle_box = Rectangle(width=2.0, height=2.5, color=BLUE, stroke_width=4)
        bicycle_box.shift(RIGHT * 2.5 + DOWN * 0.3)
        bicycle_label = Text("Bicycle (0.78)", color=BLUE).scale(0.6)
        bicycle_label.next_to(bicycle_box, UP, buff=0.15)

        self.play(
            Create(dog_box), Write(dog_label), Create(bicycle_box), Write(bicycle_label)
        )
        self.wait(1)

        # --- 3. Final title card ---
        self.play(
            FadeOut(title_clean),
            FadeOut(image),
            FadeOut(dog_box),
            FadeOut(dog_label),
            FadeOut(bicycle_box),
            FadeOut(bicycle_label),
        )

        final_title = Text("YOLOv1: You Only Look Once", color=YELLOW).scale(1.2)
        subtitle = (
            Text("Real-time Object Detection").scale(0.7).next_to(final_title, DOWN)
        )

        self.play(Write(final_title))
        self.play(Write(subtitle))
        self.wait(3)
