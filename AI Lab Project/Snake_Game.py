import os
import cv2
import mediapipe as mp
import numpy as np
import random
from collections import deque
import tensorflow as tf

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "gesture_landmark_model_best.h5")
model = tf.keras.models.load_model(model_path)

# Snake Game Constants
GRID_SIZE = 20
WINDOW_SIZE = 640
CELL_SIZE = WINDOW_SIZE // GRID_SIZE

# Colors (BGR format)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (40, 40, 40)
DARK_GREEN = (0, 100, 0)
LIGHT_GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)

# Direction vectors
DIRECTIONS = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0)
}

# Label mapping
LABELS = ["up", "down", "left", "right"]
label_map = {label: idx for idx, label in enumerate(LABELS)}
reverse_label_map = {idx: label for label, idx in label_map.items()}

def extract_landmarks(image):
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        result = hands.process(image_rgb)
        
        if not result or not result.multi_hand_landmarks:
            return None
        
        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            wrist = landmarks.landmark[0]
            index_tip = landmarks.landmark[8]
            index_mcp = landmarks.landmark[5]
            middle_tip = landmarks.landmark[12]
            ring_tip = landmarks.landmark[16]
            pinky_tip = landmarks.landmark[20]

            index_dx = index_tip.x - index_mcp.x
            index_dy = index_tip.y - index_mcp.y
            index_angle = np.arctan2(index_dy, index_dx)

            horizontal_component = np.cos(index_angle)
            vertical_component = np.sin(index_angle)

            normalized_landmarks = []
            
            # Normalize landmarks relative to wrist
            for lm in landmarks.landmark:
                x = lm.x - wrist.x
                y = lm.y - wrist.y
                z = lm.z - wrist.z
                normalized_landmarks.extend([x, y, z])

            normalized_landmarks.extend([
                index_dx,
                index_dy,
                index_angle,
                horizontal_component,
                vertical_component,
                index_tip.x - wrist.x,
                index_tip.y - wrist.y,
                (index_tip.x + middle_tip.x + ring_tip.x + pinky_tip.x) / 4 - wrist.x,
                (index_tip.y + middle_tip.y + ring_tip.y + pinky_tip.y) / 4 - wrist.y
            ])

            return np.array(normalized_landmarks)

    return None

def get_direction_from_gesture(frame, last_direction):
    features = extract_landmarks(frame)
    
    if features is not None:

        # Reshape features for model input
        features = features.reshape(1, -1)
        
        # Get prediction
        prediction = model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction[0])
        predicted_direction = reverse_label_map[predicted_class].upper()
        
        # Prevent 180-degree turns
        if (predicted_direction == 'UP' and last_direction == 'DOWN') or \
           (predicted_direction == 'DOWN' and last_direction == 'UP') or \
           (predicted_direction == 'LEFT' and last_direction == 'RIGHT') or \
           (predicted_direction == 'RIGHT' and last_direction == 'LEFT'):
            return last_direction
            
        return predicted_direction
    return last_direction

# Function to draw the grid
def draw_grid(screen):
    for i in range(GRID_SIZE + 1):
        # Vertical lines
        cv2.line(screen, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE), GRAY, 1)

        # Horizontal lines
        cv2.line(screen, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE), GRAY, 1)

def draw_cell(screen, x, y, color, is_head=False, current_direction='RIGHT'):
    # Draw cell with rounded corners
    center_x = x * CELL_SIZE + CELL_SIZE // 2
    center_y = y * CELL_SIZE + CELL_SIZE // 2
    radius = CELL_SIZE // 2 - 2
    
    if is_head:
        # Draw head with the same color as body but slightly larger
        cv2.circle(screen, (center_x, center_y), radius + 2, color, -1)

        # Draw eyes
        eye_radius = radius // 4
        if current_direction == 'RIGHT':
            cv2.circle(screen, (center_x + radius//2, center_y - radius//2), eye_radius, BLACK, -1)
            cv2.circle(screen, (center_x + radius//2, center_y + radius//2), eye_radius, BLACK, -1)
        elif current_direction == 'LEFT':
            cv2.circle(screen, (center_x - radius//2, center_y - radius//2), eye_radius, BLACK, -1)
            cv2.circle(screen, (center_x - radius//2, center_y + radius//2), eye_radius, BLACK, -1)
        elif current_direction == 'UP':
            cv2.circle(screen, (center_x - radius//2, center_y - radius//2), eye_radius, BLACK, -1)
            cv2.circle(screen, (center_x + radius//2, center_y - radius//2), eye_radius, BLACK, -1)
        else:  # DOWN
            cv2.circle(screen, (center_x - radius//2, center_y + radius//2), eye_radius, BLACK, -1)
            cv2.circle(screen, (center_x + radius//2, center_y + radius//2), eye_radius, BLACK, -1)
    else:
        cv2.circle(screen, (center_x, center_y), radius, color, -1)

def draw_food(screen, x, y):
    center_x = x * CELL_SIZE + CELL_SIZE // 2
    center_y = y * CELL_SIZE + CELL_SIZE // 2
    radius = CELL_SIZE // 2 - 2
    
    # Draw apple-like food
    cv2.circle(screen, (center_x, center_y), radius, RED, -1)

    # Add a small stem
    cv2.line(screen, (center_x, center_y - radius), 
             (center_x, center_y - radius - 5), DARK_GREEN, 2)

def draw_score(screen, score):

    # Create a semi-transparent overlay for the score
    overlay = screen.copy()
    cv2.rectangle(overlay, (10, 10), (200, 60), BLACK, -1)
    cv2.addWeighted(overlay, 0.7, screen, 0.3, 0, screen)
    
    # Draw score text
    cv2.putText(screen, f"Score: {score}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

def draw_direction(screen, direction):
    # Create a semi-transparent overlay for the direction
    overlay = screen.copy()
    cv2.rectangle(overlay, (10, 70), (200, 120), BLACK, -1)
    cv2.addWeighted(overlay, 0.7, screen, 0.3, 0, screen)
    
    # Draw direction text
    cv2.putText(screen, f"Direction: {direction}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

def random_food(snake):
    while True:
        pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if pos not in snake:
            return pos

def main():
    # Initialize game state
    snake = deque([(10, 10)])
    direction = 'RIGHT'
    food = random_food(snake)
    score = 0
    move_delay = 5
    frame_count = 0

    # Camera setup
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        # Get direction from gesture
        new_direction = get_direction_from_gesture(frame, direction)
        # Only update direction if it's not an opposite direction
        if new_direction != direction:
            direction = new_direction

        # Game logic
        frame_count += 1
        if frame_count >= move_delay:
            frame_count = 0
            dx, dy = DIRECTIONS[direction]
            head = (snake[0][0] + dx, snake[0][1] + dy)

            # Check collisions
            if (
                head in snake
                or head[0] < 0 or head[0] >= GRID_SIZE
                or head[1] < 0 or head[1] >= GRID_SIZE
            ):
                print("Game Over! Score:", score)
                break

            snake.appendleft(head)
            if head == food:
                score += 1
                food = random_food(snake)
            else:
                snake.pop()

        # Draw the game
        game_screen = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
        
        # Draw grid
        draw_grid(game_screen)
        
        # Draw snake
        for i, (x, y) in enumerate(snake):
            is_head = (i == 0)
            draw_cell(game_screen, x, y, LIGHT_GREEN, is_head, direction)
        
        # Draw food
        draw_food(game_screen, food[0], food[1])
        
        # Draw score and direction
        draw_score(game_screen, score)
        draw_direction(game_screen, direction)

        # Resize camera frame to match game screen height
        camera_frame = cv2.resize(frame, (WINDOW_SIZE, WINDOW_SIZE))
        
        # Add a border to the camera frame
        camera_frame = cv2.copyMakeBorder(camera_frame, 1, 1, 1, 1, 
                                        cv2.BORDER_CONSTANT, value=WHITE)

        # Ensure both screens have the same dimensions
        if game_screen.shape != camera_frame.shape:
            camera_frame = cv2.resize(camera_frame, (WINDOW_SIZE, WINDOW_SIZE))

        # Combine game and camera screens side by side
        combined_screen = np.hstack((game_screen, camera_frame))

        cv2.imshow("Snake Game - Gesture Controlled", combined_screen)
        
        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()