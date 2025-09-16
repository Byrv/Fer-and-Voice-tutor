import time 
class AppState:
    def __init__(self):
        self.interrupt_flag = False
        self.emotion_state = ""
        self.last_confusion_time = 0
        self.pause_tutor = False

    def trigger_interrupt(self, emotion: str):
        now = time.time()
        if emotion == "confused":
            if now - self.last_confusion_time >= 5:  # 5s cooldown
                self.last_confusion_time = now
                self.interrupt_flag = True
                self.emotion_state = emotion
        else:
            self.interrupt_flag = True
            self.emotion_state = emotion

    def clear_interrupt(self):
        # logging.info("🧹 Clearing state → interrupt_flag=False, emotion_state=''")
        self.interrupt_flag = False
        self.emotion_state = ""
