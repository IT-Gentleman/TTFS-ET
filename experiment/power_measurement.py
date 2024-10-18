import time
import threading
from jtop import jtop
import subprocess
import csv
import sys
import os

def monitor_power_usage_jtop(interval, stop_flag, power_log):
    """jtop을 사용해 실시간 전력 소비량을 샘플링"""
    with jtop() as jetson:
        prev_time = time.time()
        while not stop_flag.is_set():
            if jetson.ok():
                current_time = time.time()
                elapsed_time = current_time - prev_time
                power_data = jetson.power  # 전력 관련 데이터 수집
                total_power = power_data["tot"]["power"]  # 총 전력 소모량 (mW)
                power_log.append((elapsed_time, total_power))
                prev_time = current_time
                new_interval = interval - (time.time() - current_time)  # 샘플링 간격 유지 (time.time() - current_time은 실행 시간)
                time.sleep(new_interval)

def measure_energy_consumption_and_run_program(cmd, sampling_interval=0.1):
    """프로그램을 실행하고 그동안 전력 소모량을 측정"""
    stop_flag = threading.Event()  # 측정을 중단할 플래그
    power_log = []  # 전력 소모량 기록

    # 전력 측정 스레드 시작
    power_thread = threading.Thread(target=monitor_power_usage_jtop, args=(sampling_interval, stop_flag, power_log))
    power_thread.start()

    # 프로그램 실행 (subprocess를 이용)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 실시간으로 출력 처리
    while process.poll() is None:  # 프로세스가 종료되지 않았을 때
        output = process.stdout.readline()
        if output:
            print(output.strip())  # 실시간 출력

    # 에러 출력 처리 (프로세스가 종료된 후 남은 내용 출력)
    stdout, stderr = process.communicate()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)

    # 전력 측정 중단
    stop_flag.set()
    power_thread.join()

    # 에너지 소비량 계산 (mJ -> J)
    total_energy_mj = sum([power * elapsed for elapsed, power in power_log])
    total_energy_joules = total_energy_mj / 1000

    return total_energy_joules, power_log

def save_record_to_csv(path: str, record: list):
    file_exists = os.path.isfile(path)
    with open(path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            header = record.keys()
            writer.writerow(header)
        writer.writerow(record.values())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 본코드.py \"실행할 명령어\"")
        sys.exit(1)

    # 명령어를 인자로 받아 실행
    cmd = sys.argv[1]
    cmd = cmd.split()

    # 전력 소모량을 측정하면서 프로그램 실행
    print(f"실행 명령: {' '.join(cmd)}")
    total_energy, power_samples = measure_energy_consumption_and_run_program(cmd)

    # 결과 출력
    print(f"총 전력 소비량: {total_energy:.6f} Joules")

    # 결과를 CSV 파일로 저장
    record = {"Command": ' '.join(cmd), "Total Energy (J)": total_energy}
    save_record_to_csv("power_measurement_results.csv", record)