import pygame
import time

# Khởi tạo pygame mixer
pygame.mixer.init()

# Đường dẫn đến file âm thanh (sử dụng file của bạn)
audio_file = 'data/assets/audio/dung_len.wav'

# Chức năng phát âm thanh và đo độ trễ
def play_audio_with_latency_check(audio_file):
    # Ghi thời gian bắt đầu
    start_time = time.perf_counter()

    # Tải và phát âm thanh
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    # Chờ đến khi âm thanh thực sự bắt đầu
    while pygame.mixer.music.get_busy(): 
        pass  # Chờ cho đến khi âm thanh bắt đầu phát

    # Ghi thời gian khi âm thanh bắt đầu phát
    end_time = time.perf_counter()

    # Tính độ trễ
    latency = end_time - start_time
    print(f"Độ trễ khi phát âm thanh là: {latency:.5f} giây")

# Gọi chức năng
play_audio_with_latency_check(audio_file)