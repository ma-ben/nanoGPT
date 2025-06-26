from pathlib import Path
import time
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def log(self, step, loss, time, verbose=True):
        msg = f"Step {step:04d} | loss: {loss:.4f} | time: {time:.2f}s"
        if verbose:
            print(msg)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

def plot_time(file_path=None):
    if file_path is None:
        return
    else:
        file = Path(file_path)
    steps, times = [], []
    with file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                parts = line.strip().split('|')
                step = int(parts[0].split()[1])
                time = float(parts[2].split()[1][:-1])  # 去掉末尾的 's'
                if time > 0.2 and time <1:
                    steps.append(step)
                    times.append(time)
            except:
                continue  # 忽略非法行
    if not steps:
        print("日志为空或格式错误")
        return
    plt.plot(steps, times, label="times")
    plt.xlabel("Step")
    plt.ylabel("Time")
    plt.title("Training Time")
    plt.grid(True)
    plt.legend()
    # plt.show()
    print(len(steps))
    plt.savefig("time.png")


def plot_loss(file_path=None):
    if file_path is None:
        return
    else:
        file = Path(file_path)
    steps, losses = [], []
    with file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                parts = line.strip().split('|')
                step = int(parts[0].split()[1])
                loss = float(parts[1].split()[1])
                steps.append(step)
                losses.append(loss)
            except:
                continue  # 忽略非法行
    if not steps:
        print("日志为空或格式错误")
        return
    plt.plot(steps, losses, label="loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig("loss.png")

if __name__ == "__main__":
    log_path = "/home/taom/nanoGPT/gpt_small_checkpoints_tinysp/log.txt"
    plot_loss(log_path)
    # plot_time(log_path)