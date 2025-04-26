import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pygame
import time
import os
import json
import random

# Initialize pygame for sound
pygame.init()
pygame.mixer.init()

# Create sound effects using simple beeps
class GameSounds:
    def __init__(self):
        self.paddle_hit_sound = pygame.mixer.Sound(np.array([4096 * np.sin(2.0 * np.pi * 440.0 * x / 44100.0) for x in range(4410)]).astype(np.int16))
        self.score_sound = pygame.mixer.Sound(np.array([4096 * np.sin(2.0 * np.pi * 880.0 * x / 44100.0) for x in range(4410)]).astype(np.int16))
        self.game_over_sound = pygame.mixer.Sound(np.array([4096 * np.sin(2.0 * np.pi * 220.0 * x / 44100.0) for x in range(8820)]).astype(np.int16))
        
        # Set even lower volume
        self.paddle_hit_sound.set_volume(0.2)
        self.score_sound.set_volume(0.2)
        self.game_over_sound.set_volume(0.2)
        
        # Add cooldown system for sounds
        self.last_hit_time = 0
        self.sound_cooldown = 0.2  # 200ms cooldown between sounds
    
    def paddle_hit(self):
        current_time = time.time()
        if current_time - self.last_hit_time > self.sound_cooldown:
            try:
                self.paddle_hit_sound.play()
                self.last_hit_time = current_time
            except:
                pass
        
    def score(self):
        current_time = time.time()
        if current_time - self.last_hit_time > self.sound_cooldown:
            try:
                self.score_sound.play()
                self.last_hit_time = current_time
            except:
                pass
        
    def game_over(self):
        current_time = time.time()
        if current_time - self.last_hit_time > self.sound_cooldown:
            try:
                self.game_over_sound.play()
                self.last_hit_time = current_time
            except:
                pass

# Game settings
class GameSettings:
    def __init__(self):
        self.difficulty = 1  # 1: Easy, 2: Medium, 3: Hard
        self.initial_speeds = {1: 10, 2: 15, 3: 20}
        self.max_speeds = {1: 20, 2: 25, 3: 30}
        self.paddle_sizes = {1: 1.0, 2: 0.8, 3: 0.6}  # Scale factor for paddle size
        self.speed_increase_interval = 5  # Increase speed every 5 points
        self.game_mode = GameMode.NORMAL
        self.ai_enabled = False
        self.time_attack_duration = 60  # seconds
        self.shield_active = False

class GameMode:
    NORMAL = "normal"
    TIME_ATTACK = "time_attack"
    SURVIVAL = "survival"
    PRACTICE = "practice"
    AI = "ai"

class AIPlayer:
    def __init__(self, difficulty):
        self.difficulty = difficulty
        self.reaction_speeds = {1: 0.3, 2: 0.6, 3: 0.9}  # How quickly AI reacts
        self.prediction_error = {1: 100, 2: 50, 3: 20}  # Pixels of random error
        self.target_y = 360  # Middle of screen
        
    def update(self, ball_pos, ball_speed):
        # Predict where ball will intersect with paddle
        if (ball_speed[0] > 0):  # Ball moving towards AI
            time_to_intercept = (1195 - ball_pos[0]) / ball_speed[0]
            predicted_y = ball_pos[1] + ball_speed[1] * time_to_intercept
            # Add some prediction error based on difficulty
            predicted_y += random.randint(-self.prediction_error[self.difficulty], 
                                       self.prediction_error[self.difficulty])
            # Move towards prediction with speed based on difficulty
            self.target_y += (predicted_y - self.target_y) * self.reaction_speeds[self.difficulty]
            return int(np.clip(self.target_y, 20, 415))
        return self.target_y

class PowerUp:
    def __init__(self):
        self.active = False
        self.position = [0, 0]
        self.type = None
        self.duration = 0
        self.start_time = 0
        self.types = {
            'big_paddle': {'duration': 5, 'color': (0, 255, 0)},
            'slow_ball': {'duration': 3, 'color': (0, 0, 255)},
            'double_points': {'duration': 4, 'color': (255, 255, 0)},
            'shield': {'duration': 5, 'color': (128, 128, 255)},
            'invert_controls': {'duration': 3, 'color': (255, 128, 0)}
        }
    
    def spawn(self):
        if not self.active:
            self.position = [random.randint(100, 1180), random.randint(100, 620)]
            self.type = random.choice(list(self.types.keys()))
            self.duration = self.types[self.type]['duration']
            self.start_time = time.time()  # Initialize time when spawning
            self.active = True
    
    def draw(self, img):
        if self.active:
            color = self.types[self.type]['color']
            cv2.circle(img, (int(self.position[0]), int(self.position[1])), 15, color, -1)
            cv2.circle(img, (int(self.position[0]), int(self.position[1])), 17, (255, 255, 255), 2)

class HighScores:
    def __init__(self):
        self.scores_file = "highscores.json"
        self.scores = {
            'easy': [],
            'medium': [],
            'hard': []
        }
        self.load_scores()
    
    def load_scores(self):
        try:
            if os.path.exists(self.scores_file):
                with open(self.scores_file, 'r') as f:
                    self.scores = json.load(f)
        except:
            pass
    
    def save_scores(self):
        try:
            with open(self.scores_file, 'w') as f:
                json.dump(self.scores, f)
        except:
            pass
    
    def add_score(self, difficulty, score):
        diff_name = ['easy', 'medium', 'hard'][difficulty-1]
        self.scores[diff_name].append(score)
        self.scores[diff_name].sort(reverse=True)
        self.scores[diff_name] = self.scores[diff_name][:5]  # Keep top 5
        self.save_scores()
    
    def get_high_score(self, difficulty):
        diff_name = ['easy', 'medium', 'hard'][difficulty-1]
        return max(self.scores[diff_name]) if self.scores[diff_name] else 0

class BallTrail:
    def __init__(self, max_length=10):
        self.positions = []
        self.max_length = max_length
    
    def update(self, pos):
        self.positions.append(pos.copy())
        if len(self.positions) > self.max_length:
            self.positions.pop(0)
    
    def draw(self, img):
        for i in range(len(self.positions)-1):
            alpha = (i + 1) / len(self.positions)
            radius = int(10 * alpha)
            cv2.circle(img, 
                      (int(self.positions[i][0]), int(self.positions[i][1])), 
                      radius, (255, 255, 255), 
                      1)

class ScoreAnimation:
    def __init__(self):
        self.active = False
        self.start_time = 0
        self.position = [0, 0]
        self.score = 0
        self.duration = 1.0  # seconds
        
    def start(self, position, score):
        self.active = True
        self.start_time = time.time()
        self.position = position.copy()
        self.score = score
    
    def draw(self, img):
        if self.active:
            elapsed = time.time() - self.start_time
            if elapsed > self.duration:
                self.active = False
                return
            
            # Move animation upward
            y_offset = int(50 * elapsed)
            # Fade out text
            alpha = 1.0 - (elapsed / self.duration)
            
            # Draw score with fade effect
            pos = (int(self.position[0]), int(self.position[1] - y_offset))
            cv2.putText(img, f"+{self.score}", pos,
                       cv2.FONT_HERSHEY_COMPLEX, 1,
                       (255, 255, 255), 2, cv2.LINE_AA)

class Ball:
    def __init__(self, pos, speed_x, speed_y):
        self.pos = list(pos)
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.trail = BallTrail(max_length=8)
        self.wall_bounce_cooldown = 0
        self.wall_cooldown_time = 0.1  # 100ms cooldown
        self.last_collision_time = 0
        self.collision_cooldown = 0.2  # 200ms between collisions
        self.previous_pos = list(pos)  # Store previous position for better collision
        
    def update(self, speed_multiplier, max_speed):
        current_time = time.time()
        # Store previous position
        self.previous_pos = self.pos.copy()
        
        # Calculate new position first
        current_speed_x = np.clip(self.speed_x * speed_multiplier, -max_speed, max_speed)
        current_speed_y = np.clip(self.speed_y * speed_multiplier, -max_speed, max_speed)
        new_pos_x = self.pos[0] + current_speed_x
        new_pos_y = self.pos[1] + current_speed_y
        
        # Handle wall collisions more smoothly with cooldown
        if (new_pos_y >= 500 or new_pos_y <= 10) and current_time - self.last_collision_time > self.collision_cooldown:
            self.speed_y = -self.speed_y
            self.last_collision_time = current_time
            # Adjust position to prevent sticking to wall
            new_pos_y = np.clip(new_pos_y, 11, 499)
        
        # Update position with bounds checking
        self.pos[0] = int(np.clip(new_pos_x, 40, 1240))
        self.pos[1] = int(new_pos_y)
        
        # Update trail
        self.trail.update(self.pos)

    def check_paddle_collision(self, paddle_x, paddle_width, paddle_y, paddle_height):
        # Check if ball is moving towards the paddle
        if (paddle_x < 640 and self.speed_x < 0) or (paddle_x > 640 and self.speed_x > 0):
            # Create a slightly larger hit box for more forgiving collision
            hit_box_width = paddle_width + 10
            hit_box_x = paddle_x - 5 if paddle_x < 640 else paddle_x
            
            # Check current and predicted position for collision
            current_collision = (hit_box_x <= self.pos[0] <= hit_box_x + hit_box_width and
                               paddle_y <= self.pos[1] <= paddle_y + paddle_height)
            
            # Add trajectory-based collision for better detection
            if not current_collision and self.previous_pos[0] != self.pos[0]:
                # Calculate intersection point with paddle plane
                t = (hit_box_x - self.previous_pos[0]) / (self.pos[0] - self.previous_pos[0])
                if 0 <= t <= 1:  # Check if intersection happens during this frame
                    intersect_y = self.previous_pos[1] + t * (self.pos[1] - self.previous_pos[1])
                    if paddle_y <= intersect_y <= paddle_y + paddle_height:
                        return True, intersect_y
            
            return current_collision, self.pos[1]
        return False, self.pos[1]

    def draw(self, img):
        self.trail.draw(img)
        return cvzone.overlayPNG(img, imgBall, self.pos)

class Achievement:
    def __init__(self, name, description, requirement):
        self.name = name
        self.description = description
        self.requirement = requirement
        self.unlocked = False
        self.just_unlocked = False

class AchievementSystem:
    def __init__(self):
        self.achievements_file = "achievements.json"
        self.achievements = {
            'score_master': Achievement('Score Master', 'Score 50 points in total', 50),
            'power_collector': Achievement('Power Collector', 'Collect 10 power-ups', 10),
            'speed_demon': Achievement('Speed Demon', 'Reach 2x speed multiplier', 2.0),
            'survivor': Achievement('Survivor', 'Play for 3 minutes in survival mode', 180),
            'ai_beater': Achievement('AI Beater', 'Beat the AI in hard mode', 1),
            'shield_master': Achievement('Shield Master', 'Use 5 shields successfully', 5)
        }
        self.stats = {
            'total_score': 0,
            'power_ups_collected': 0,
            'max_speed': 1.0,
            'survival_time': 0,
            'ai_wins': 0,
            'shields_used': 0
        }
        self.load_achievements()
    
    def load_achievements(self):
        try:
            if os.path.exists(self.achievements_file):
                with open(self.achievements_file, 'r') as f:
                    data = json.load(f)
                    self.stats = data['stats']
                    for k, v in data['achievements'].items():
                        self.achievements[k].unlocked = v
        except:
            pass
    
    def save_achievements(self):
        try:
            with open(self.achievements_file, 'w') as f:
                json.dump({
                    'stats': self.stats,
                    'achievements': {k: v.unlocked for k, v in self.achievements.items()}
                }, f)
        except:
            pass
    
    def update(self, stats_updates):
        for stat, value in stats_updates.items():
            if stat in self.stats:
                self.stats[stat] = max(self.stats[stat], value)
        
        # Check achievements
        achievements = self.achievements
        if not achievements['score_master'].unlocked and self.stats['total_score'] >= achievements['score_master'].requirement:
            achievements['score_master'].unlocked = True
            achievements['score_master'].just_unlocked = True
            
        if not achievements['power_collector'].unlocked and self.stats['power_ups_collected'] >= achievements['power_collector'].requirement:
            achievements['power_collector'].unlocked = True
            achievements['power_collector'].just_unlocked = True
            
        if not achievements['speed_demon'].unlocked and self.stats['max_speed'] >= achievements['speed_demon'].requirement:
            achievements['speed_demon'].unlocked = True
            achievements['speed_demon'].just_unlocked = True
            
        if not achievements['survivor'].unlocked and self.stats['survival_time'] >= achievements['survivor'].requirement:
            achievements['survivor'].unlocked = True
            achievements['survivor'].just_unlocked = True
            
        if not achievements['ai_beater'].unlocked and self.stats['ai_wins'] >= achievements['ai_beater'].requirement:
            achievements['ai_beater'].unlocked = True
            achievements['ai_beater'].just_unlocked = True
            
        if not achievements['shield_master'].unlocked and self.stats['shields_used'] >= achievements['shield_master'].requirement:
            achievements['shield_master'].unlocked = True
            achievements['shield_master'].just_unlocked = True
        
        self.save_achievements()
    
    def draw_notifications(self, img):
        for achievement in self.achievements.values():
            if achievement.just_unlocked:
                # Draw achievement notification
                cv2.rectangle(img, (400, 300), (880, 380), (0, 255, 0), -1)
                cv2.rectangle(img, (400, 300), (880, 380), (255, 255, 255), 2)
                cv2.putText(img, "Achievement Unlocked!", (420, 330),
                           cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, achievement.name, (420, 360),
                           cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                achievement.just_unlocked = False

class Particle:
    def __init__(self, pos, velocity, color, life=1.0, size=3):
        self.pos = list(pos)
        self.velocity = list(velocity)
        self.color = color
        self.life = life
        self.size = size
        self.alpha = 1.0
        
    def update(self):
        self.pos[0] += self.velocity[0]
        self.pos[1] += self.velocity[1]
        self.life -= 0.016  # Assume 60 FPS
        self.alpha = self.life
        return self.life > 0
        
    def draw(self, img):
        if self.life > 0:
            cv2.circle(img, 
                      (int(self.pos[0]), int(self.pos[1])), 
                      self.size,
                      self.color,
                      -1)

class ParticleSystem:
    def __init__(self):
        self.particles = []
        
    def emit(self, pos, direction, color, count=10):
        for _ in range(count):
            angle = np.random.uniform(-0.5, 0.5) + (0 if direction[0] > 0 else np.pi)
            speed = np.random.uniform(5, 15)
            velocity = [speed * np.cos(angle), speed * np.sin(angle)]
            life = np.random.uniform(0.3, 0.8)
            size = np.random.randint(2, 5)
            self.particles.append(Particle(pos, velocity, color, life, size))
    
    def update_and_draw(self, img):
        self.particles = [p for p in self.particles if p.update()]
        for p in self.particles:
            p.draw(img)

class ComboSystem:
    def __init__(self):
        self.combo = 0
        self.max_combo = 0
        self.combo_timer = 0
        self.combo_timeout = 2.0  # Seconds before combo resets
        self.points_multiplier = 1.0
        self.shake_amount = 0
        
    def hit(self):
        self.combo += 1
        self.max_combo = max(self.max_combo, self.combo)
        self.combo_timer = time.time()
        self.points_multiplier = 1.0 + (self.combo * 0.1)  # 10% increase per combo
        self.shake_amount = min(self.combo * 2, 20)
        
    def miss(self):
        self.combo = 0
        self.points_multiplier = 1.0
        self.shake_amount = 0
        
    def update(self):
        if self.combo > 0 and time.time() - self.combo_timer > self.combo_timeout:
            self.miss()
        
    def draw(self, img):
        if self.combo > 0:
            # Draw combo counter with pulse effect
            scale = 1.0 + 0.2 * np.sin(time.time() * 10)
            size = int(30 * scale)
            cv2.putText(img, f"Combo: {self.combo}x", (50, 50),
                       cv2.FONT_HERSHEY_COMPLEX, size/30,
                       (255, 255, 255), 2)
            
            # Add screen shake
            if self.shake_amount > 0:
                dx = int(np.random.uniform(-1, 1) * self.shake_amount)
                dy = int(np.random.uniform(-1, 1) * self.shake_amount)
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                self.shake_amount *= 0.9  # Decay shake

class PaddleAbility:
    def __init__(self):
        self.charge = 0
        self.max_charge = 100
        self.is_active = False
        self.duration = 0
        self.start_time = None
        
    def add_charge(self, amount):
        self.charge = min(self.max_charge, self.charge + amount)
        
    def activate(self, duration):
        if self.charge >= self.max_charge:
            self.is_active = True
            self.duration = duration
            self.start_time = time.time()
            self.charge = 0
            return True
        return False
        
    def update(self):
        if self.is_active and time.time() - self.start_time > self.duration:
            self.is_active = False
            
    def draw(self, img, x, y):
        # Draw charge bar
        bar_width = 100
        bar_height = 10
        filled_width = int(bar_width * (self.charge / self.max_charge))
        cv2.rectangle(img, (x, y), (x + bar_width, y + bar_height), (128, 128, 128), -1)
        if filled_width > 0:
            cv2.rectangle(img, (x, y), (x + filled_width, y + bar_height), 
                         (0, 255, 0) if self.charge >= self.max_charge else (0, 128, 255), -1)

class GameMusic:
    def __init__(self):
        # Create a continuous low hum that changes with game intensity
        sample_rate = 44100
        duration = 2.0  # 2 second loop
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Base tone (low hum)
        base_freq = 100
        self.base_sound = np.sin(2 * np.pi * base_freq * t) * 0.3
        
        # Higher frequency for intensity
        high_freq = 200
        self.high_sound = np.sin(2 * np.pi * high_freq * t) * 0.2
        
        # Convert to 16-bit integer format
        self.base_sound = (self.base_sound * 32767).astype(np.int16)
        self.high_sound = (self.high_sound * 32767).astype(np.int16)
        
        # Create pygame Sound objects
        self.base_channel = pygame.mixer.Channel(0)
        self.high_channel = pygame.mixer.Channel(1)
        self.base_sound_obj = pygame.mixer.Sound(self.base_sound)
        self.high_sound_obj = pygame.mixer.Sound(self.high_sound)
        
        # Set initial volumes
        self.base_channel.set_volume(0.1)
        self.high_channel.set_volume(0.0)
        
    def play(self):
        self.base_channel.play(self.base_sound_obj, -1)  # -1 for looping
        self.high_channel.play(self.high_sound_obj, -1)
        
    def update_intensity(self, intensity):
        # Lower the intensity effect
        self.high_channel.set_volume(min(intensity * 0.1, 0.1))
        
    def stop(self):
        self.base_channel.stop()
        self.high_channel.stop()

# Initialize game components
sounds = GameSounds()
settings = GameSettings()
power_up = PowerUp()
high_scores = HighScores()
achievements = AchievementSystem()
particle_system = ParticleSystem()
combo_system = ComboSystem()
left_ability = PaddleAbility()
right_ability = PaddleAbility()
game_music = GameMusic()
game_music.play()  # Start background music

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Importing all images
imgBackground = cv2.imread("Resources/Background.png")
imgGameOver = cv2.imread("Resources/gameOver.png")
imgBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread("Resources/bat1.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("Resources/bat2.png", cv2.IMREAD_UNCHANGED)

# Check if images loaded successfully
if imgBackground is None or imgGameOver is None or imgBall is None or imgBat1 is None or imgBat2 is None:
    print("Error: Failed to load one or more game images")
    exit(1)

# Resize background to match camera resolution if needed
imgBackground = cv2.resize(imgBackground, (1280, 720))

# Create named window
cv2.namedWindow('Pong Game', cv2.WINDOW_NORMAL)

# Hand Detector
try:
    detector = HandDetector(detectionCon=0.8, maxHands=2)
except Exception as e:
    print(f"Error initializing hand detector: {e}")
    exit(1)

# Check if camera is available
if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

# Variables
speedX = int(settings.initial_speeds[settings.difficulty])
speedY = int(settings.initial_speeds[settings.difficulty])
ball = Ball([100, 100], speedX, speedY)
gameOver = False
score = [0, 0]
paused = False
countdown = 3  # Start with countdown
paddle_scale = float(settings.paddle_sizes[settings.difficulty])
active_power_ups = {}
ai_player = None
game_start_time = time.time()
shields_remaining = 0

# Initialize visual effects
ball_trail = BallTrail(max_length=8)
score_animation = ScoreAnimation()

def reset_game():
    global ball, speedX, speedY, gameOver, score, countdown, active_power_ups, shields_remaining, game_start_time
    ball = Ball([100, 100], 
                int(settings.initial_speeds[settings.difficulty]), 
                int(settings.initial_speeds[settings.difficulty]))
    gameOver = False
    score = [0, 0]
    countdown = 3
    active_power_ups = {}
    shields_remaining = 0
    game_start_time = time.time()  # Always initialize time on reset
    power_up.start_time = time.time()  # Reset power-up timer
    if settings.game_mode == GameMode.TIME_ATTACK:
        game_start_time = time.time()
    elif settings.game_mode == GameMode.AI:
        global ai_player
        ai_player = AIPlayer(settings.difficulty)
    combo_system.miss()  # Reset combo

def draw_countdown(img, number):
    h, w = img.shape[:2]
    cv2.putText(img, str(number), (w//2 - 50, h//2), 
                cv2.FONT_HERSHEY_COMPLEX, 5, (255, 255, 255), 5)

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame from camera")
        break
        
    try:
        img = cv2.flip(img, 1)
        imgRaw = img.copy()

        # Find the hand and its landmarks
        hands, img = detector.findHands(img, flipType=False)

        # Overlaying the background image
        img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

        if paused:
            cv2.putText(img, "PAUSED", (500, 260), cv2.FONT_HERSHEY_COMPLEX, 
                        3, (255, 255, 255), 5)
            y_offset = 320
            for achievement in achievements.achievements.values():
                status = "✓" if achievement.unlocked else "✗"
                cv2.putText(img, f"{status} {achievement.name}: {achievement.description}", 
                           (300, y_offset), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                           (255, 255, 255) if achievement.unlocked else (128, 128, 128), 2)
                y_offset += 30
        elif countdown > 0:
            current_time = time.time()
            draw_countdown(img, int(countdown))
            countdown = max(0, countdown - 0.016)  # Decrease countdown (assumes 60 FPS)
        else:
            # Check for hands
            if hands:
                for hand in hands:
                    x, y, w, h = map(int, hand['bbox'])  # Convert to integers
                    h1, w1, _ = imgBat1.shape
                    h1 = int(h1 * paddle_scale)  # Scale paddle height
                    w1 = int(w1)  # Ensure width is integer
                    y1 = int(y - h1 // 2)
                    y1 = int(np.clip(y1, 20, 415))

                    if hand['type'] == "Left":
                        img = cvzone.overlayPNG(img, cv2.resize(imgBat1, (w1, h1)), (59, y1))
                        # Check collision with enhanced detection
                        hit, contact_y = ball.check_paddle_collision(59, w1, y1, h1)
                        if hit and time.time() - ball.last_collision_time > ball.collision_cooldown:
                            ball.speed_x = abs(ball.speed_x)  # Ensure ball moves right
                            ball.pos[0] = 60 + w1  # Move ball outside paddle
                            ball.last_collision_time = time.time()
                            
                            # Adjust ball angle based on where it hits the paddle
                            relative_intersect_y = (contact_y - y1) / h1
                            bounce_angle = relative_intersect_y * 0.5  # Max 45-degree bounce
                            speed = np.sqrt(ball.speed_x**2 + ball.speed_y**2)
                            ball.speed_y = speed * np.sin(bounce_angle)
                            ball.speed_x = speed * np.cos(bounce_angle)
                            
                            # Rest of hit handling code...
                            combo_system.hit()
                            points = int(2 if 'double_points' in active_power_ups else 1 * combo_system.points_multiplier)
                            score[0] += points
                            score_animation.start(ball.pos, points)
                            sounds.paddle_hit()
                            particle_system.emit(ball.pos, [1, 0], (0, 255, 0))
                            left_ability.add_charge(20)

                    if hand['type'] == "Right":
                        img = cvzone.overlayPNG(img, cv2.resize(imgBat2, (w1, h1)), (1195, y1))
                        current_time = time.time()
                        if (1195 - 50 < ball.pos[0] < 1195 and 
                            y1 <= ball.pos[1] <= y1 + h1 and
                            current_time - ball.last_collision_time > ball.collision_cooldown):
                            ball.speed_x = -ball.speed_x
                            ball.pos[0] = int(ball.pos[0] - 30)
                            ball.last_collision_time = current_time
                            combo_system.hit()
                            points = int(2 if 'double_points' in active_power_ups else 1 * combo_system.points_multiplier)
                            score[1] += points
                            score_animation.start(ball.pos, points)
                            sounds.paddle_hit()
                            particle_system.emit(ball.pos, [-1, 0], (0, 255, 0))
                            right_ability.add_charge(20)  # Add charge on successful hit
                            
                            # Special ability: Multi-hit (if activated)
                            if right_ability.is_active:
                                score[1] += 1  # Extra point
                                particle_system.emit(ball.pos, [-1, 0], (148, 0, 211), 20)  # Special particles

            # Game Over
            if ball.pos[0] < 40 or ball.pos[0] > 1200:
                if not settings.shield_active:
                    if not gameOver:
                        combo_system.miss()  # Reset combo on miss
                        sounds.game_over()
                        gameOver = True

            if gameOver:
                img = imgGameOver
                cv2.putText(img, str(score[1] + score[0]).zfill(2), (585, 360),
                            cv2.FONT_HERSHEY_COMPLEX, 2.5, (200, 0, 200), 5)
                total_score = score[0] + score[1]
                high_scores.add_score(settings.difficulty, total_score)
                high_score = high_scores.get_high_score(settings.difficulty)
                cv2.putText(img, f"High Score: {high_score}", (500, 400),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            else:
                # Move the Ball
                if ball.pos[1] >= 500 or ball.pos[1] <= 10:
                    ball.speed_y = -ball.speed_y
                
                # Increase speed based on total score
                total_score = score[0] + score[1]
                speed_multiplier = 1 + (total_score // settings.speed_increase_interval) * 0.1
                current_speed_x = ball.speed_x * speed_multiplier
                current_speed_y = ball.speed_y * speed_multiplier
                
                # Cap the speed at max_speed
                max_speed = settings.max_speeds[settings.difficulty]
                current_speed_x = np.clip(current_speed_x, -max_speed, max_speed)
                current_speed_y = np.clip(current_speed_y, -max_speed, max_speed)

                # Update ball position with bounds checking and ensure integers
                ball.pos[0] = int(np.clip(ball.pos[0] + current_speed_x, 0, 1280))
                ball.pos[1] = int(np.clip(ball.pos[1] + current_speed_y, 0, 720))

                # Draw ball trail
                ball_trail.update(ball.pos)
                ball_trail.draw(img)

                # Draw the ball
                img = cvzone.overlayPNG(img, imgBall, ball.pos)

                # Draw game info with speed multiplier
                cv2.putText(img, f"Score: {score[0]} - {score[1]}", (500, 650), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, f"Difficulty: {['Easy', 'Medium', 'Hard'][settings.difficulty-1]}", 
                            (500, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, f"Speed: x{speed_multiplier:.1f}", (500, 70), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                # Game logic
                if not gameOver:
                    # Spawn power-up with 1% chance per frame
                    if not power_up.active and random.random() < 0.01:
                        power_up.spawn()
                    
                    # Draw and check power-up collision
                    if power_up.active:
                        power_up.draw(img)
                        # Check if ball hits power-up
                        if (abs(ball.pos[0] - power_up.position[0]) < 20 and 
                            abs(ball.pos[1] - power_up.position[1]) < 20):
                            # Apply power-up effect
                            if power_up.type == 'big_paddle':
                                paddle_scale *= 1.5
                            elif power_up.type == 'slow_ball':
                                ball.speed_x *= 0.5
                                ball.speed_y *= 0.5
                            power_up.active = False
                            power_up.start_time = time.time()
                            active_power_ups[power_up.type] = power_up.duration
                    
                    # Update and draw active power-ups
                    current_time = time.time()
                    active_powers = list(active_power_ups.keys())
                    for power in active_powers:
                        if power_up.start_time is not None and current_time - power_up.start_time > active_power_ups[power]:
                            if power == 'big_paddle':
                                paddle_scale = float(settings.paddle_sizes[settings.difficulty])
                            elif power == 'slow_ball':
                                ball.speed_x *= 2
                                ball.speed_y *= 2
                            del active_power_ups[power]
                    
                    # Draw active power-ups with safety check
                    y_offset = 110
                    for power in active_power_ups:
                        if power_up.start_time is not None:
                            remaining = max(0, int(active_power_ups[power] - (current_time - power_up.start_time)))
                            cv2.putText(img, f"{power}: {remaining}s", (20, y_offset), 
                                      cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                            y_offset += 30

                # Draw score animation if active
                if score_animation.active:
                    score_animation.draw(img)

                # Handle game modes with safety checks
                if settings.game_mode == GameMode.TIME_ATTACK and game_start_time is not None:
                    time_remaining = max(0, settings.time_attack_duration - (time.time() - game_start_time))
                    if time_remaining <= 0:
                        gameOver = True
                    else:
                        cv2.putText(img, f"Time: {int(time_remaining)}s", (580, 110), 
                                   cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                
                elif settings.game_mode == GameMode.SURVIVAL and game_start_time is not None:
                    speed_multiplier = 1 + (time.time() - game_start_time) / 10  # Increase speed over time
                
                elif settings.game_mode == GameMode.AI and ai_player:
                    # Update AI paddle position
                    ai_y = ai_player.update(ball.pos, [ball.speed_x, ball.speed_y])
                    img = cvzone.overlayPNG(img, cv2.resize(imgBat2, (w1, h1)), (1195, ai_y))
                    
                    # Check for AI paddle hits
                    if 1195 - 50 < ball.pos[0] < 1195 and ai_y < ball.pos[1] < ai_y + h1:
                        ball.speed_x = -ball.speed_x
                        ball.pos[0] = int(ball.pos[0] - 30)
                        sounds.paddle_hit()

                # Update and draw ball
                ball.update(speed_multiplier, max_speed)
                img = ball.draw(img)
                
                # Check if ball is out
                if ball.pos[0] <= 40 or ball.pos[0] >= 1200:  # Changed < to <= for left wall
                    if not settings.shield_active:
                        if not gameOver:
                            sounds.game_over()
                            gameOver = True
                    else:
                        # Use shield to save the ball
                        settings.shield_active = False
                        shields_remaining -= 1
                        ball.speed_x = -ball.speed_x
                        # Move ball away from wall slightly
                        if ball.pos[0] <= 40:  # Changed < to <= for consistency
                            ball.pos[0] = 41
                        else:
                            ball.pos[0] = 1199

                # Handle paddle hits
                if hand['type'] == "Left":
                    img = cvzone.overlayPNG(img, cv2.resize(imgBat1, (w1, h1)), (59, y1))
                    # Improved left paddle collision check
                    if (59 <= ball.pos[0] <= 59 + w1 and  # Changed < to <= for more reliable detection
                        y1 <= ball.pos[1] <= y1 + h1):    # Changed < to <= for consistency
                        ball.speed_x = abs(ball.speed_x)  # Ensure ball moves right
                        ball.pos[0] = 60 + w1  # Move ball outside paddle
                        combo_system.hit()
                        points = int(2 if 'double_points' in active_power_ups else 1 * combo_system.points_multiplier)
                        score[0] += points
                        score_animation.start(ball.pos, points)
                        sounds.paddle_hit()
                        particle_system.emit(ball.pos, [1, 0], (0, 255, 0))
                        left_ability.add_charge(20)  # Add charge on successful hit
                        
                        # Special ability: Super Speed (if activated)
                        if left_ability.is_active:
                            ball.speed_x *= 1.5  # 50% faster
                            particle_system.emit(ball.pos, [1, 0], (255, 165, 0), 20)  # Special particles

                if hand['type'] == "Right":
                    img = cvzone.overlayPNG(img, cv2.resize(imgBat2, (w1, h1)), (1195, y1))
                    if 1195 - 50 < ball.pos[0] < 1195 and y1 < ball.pos[1] < y1 + h1:
                        ball.speed_x = -ball.speed_x
                        ball.pos[0] = int(ball.pos[0] - 30)
                        combo_system.hit()
                        points = int(2 if 'double_points' in active_power_ups else 1 * combo_system.points_multiplier)
                        score[1] += points
                        score_animation.start(ball.pos, points)
                        sounds.paddle_hit()
                        particle_system.emit(ball.pos, [-1, 0], (0, 255, 0))
                        right_ability.add_charge(20)  # Add charge on successful hit
                        
                        # Special ability: Multi-hit (if activated)
                        if right_ability.is_active:
                            score[1] += 1  # Extra point
                            particle_system.emit(ball.pos, [-1, 0], (148, 0, 211), 20)  # Special particles

                # AI paddle hits
                elif settings.game_mode == GameMode.AI and ai_player:
                    # Update AI paddle position
                    ai_y = ai_player.update(ball.pos, [ball.speed_x, ball.speed_y])
                    img = cvzone.overlayPNG(img, cv2.resize(imgBat2, (w1, h1)), (1195, ai_y))
                    
                    # Check for AI paddle hits
                    if 1195 - 50 < ball.pos[0] < 1195 and ai_y < ball.pos[1] < ai_y + h1:
                        ball.speed_x = -ball.speed_x
                        ball.pos[0] = int(ball.pos[0] - 30)
                        sounds.paddle_hit()

                # Handle power-up effects
                if power_up.type == 'shield':
                    settings.shield_active = True
                    shields_remaining += 1
                    
                # Draw shield indicator if active
                if shields_remaining > 0:
                    cv2.putText(img, f"Shields: {shields_remaining}", (20, 180), 
                               cv2.FONT_HERSHEY_COMPLEX, 0.7, (128, 128, 255), 2)

                # Update abilities
                left_ability.update()
                right_ability.update()
                
                # Draw ability charge bars
                left_ability.draw(img, 59, 500)
                right_ability.draw(img, 1195, 500)
                
                # Update music intensity based on game state
                game_intensity = min((speed_multiplier - 1) * 0.5 + (combo_system.combo * 0.1), 1.0)
                game_music.update_intensity(game_intensity)

                # Update achievements
                if not gameOver:
                    current_time = time.time()
                    game_time = current_time - game_start_time
                    
                    achievements.update({
                        'total_score': score[0] + score[1],
                        'power_ups_collected': len(active_power_ups),
                        'max_speed': speed_multiplier,
                        'survival_time': game_time if settings.game_mode == GameMode.SURVIVAL else 0,
                        'shields_used': shields_remaining
                    })
                    
                    # Check for AI win
                    if settings.game_mode == GameMode.AI and settings.difficulty == 3 and score[0] > score[1]:
                        achievements.update({'ai_wins': 1})
                
                # Draw achievement notifications
                achievements.draw_notifications(img)

                # Update and draw particles
                particle_system.update_and_draw(img)
                
                # Update and draw combo system
                combo_system.update()
                combo_system.draw(img)

        # Show webcam preview
        preview_height = 120
        preview_width = 213
        img[580:580+preview_height, 20:20+preview_width] = cv2.resize(imgRaw, (preview_width, preview_height))

        # Display the game window
        cv2.imshow('Pong Game', img)

        # Handle keyboard input
        key = cv2.waitKey(1)
        if key == ord('r'):  # Reset game
            reset_game()
        elif key == ord('p'):  # Pause game
            paused = not paused
        elif key == ord('1'):  # Easy difficulty
            settings.difficulty = 1
            paddle_scale = settings.paddle_sizes[1]
            reset_game()
        elif key == ord('2'):  # Medium difficulty
            settings.difficulty = 2
            paddle_scale = settings.paddle_sizes[2]
            reset_game()
        elif key == ord('3'):  # Hard difficulty
            settings.difficulty = 3
            paddle_scale = settings.paddle_sizes[3]
            reset_game()
        elif key == ord('m'):  # Toggle game mode
            modes = [GameMode.NORMAL, GameMode.TIME_ATTACK, GameMode.SURVIVAL, GameMode.PRACTICE, GameMode.AI]
            current_index = modes.index(settings.game_mode)
            settings.game_mode = modes[(current_index + 1) % len(modes)]
            reset_game()
        elif key == ord('q'):  # Left paddle ability
            if left_ability.activate(5.0):  # 5 second duration
                particle_system.emit([100, 360], [1, 0], (255, 165, 0), 30)
        elif key == ord('e'):  # Right paddle ability
            if right_ability.activate(5.0):  # 5 second duration
                particle_system.emit([1180, 360], [-1, 0], (148, 0, 211), 30)
        elif key == 27:  # ESC to quit
            break
    except Exception as e:
        print(f"Error during game loop: {e}")
        break

cap.release()
cv2.destroyAllWindows()
game_music.stop()
pygame.quit()
