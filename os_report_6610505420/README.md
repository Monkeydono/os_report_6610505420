# OS Report 2026

- **ชื่อ-นามสกุล:** ธยศ ขันทะยศ
- **รหัสนิสิต:** 6610505420

> งานนี้ทำขึ้นเพื่อการศึกษาในรายวิชาระบบปฏิบัติการเท่านั้น ไม่ใช่โปรแกรมสำหรับนำไปใช้งานจริง
> (This project is for education only. It is not production software.)

## โครงสร้างไฟล์

```
os_report_6610505420
|-- 1_parallel_6610505420      ข้อ 1 Intro to Parallel Programming (OpenMP)
|   |-- parallel.c             โปรแกรมหลัก: คูณเมทริกซ์ n x n แบบขนาน (4 อัลกอริทึม)
|   |-- bench.py               สคริปต์รันการทดลองทั้งหมด -> ไฟล์ CSV
|   |-- plot.py                สคริปต์วาดกราฟ + คำนวณ Amdahl's law -> PNG และ summary.md
|   |-- Makefile
|   └-- SKILL.md               skill ของ AI ที่ใช้ระหว่างเขียนโค้ด (ดูหัวข้อการใช้ AI)
|-- 2_mmap_6610505420          ข้อ 2 Memory Map Visualization
|   |-- mmap.py                โปรแกรมหลัก: แสดงภาพ memory map ของโปรเซส (Python 3, standard library)
|   |-- demo.c                 โปรแกรมตัวอย่างที่ memory map เปลี่ยนเป็นขั้นๆ ไว้ใช้สาธิต
|   |-- Makefile
|   └-- SKILL.md
|-- README.md
└-- report_6610505420.pdf
```

## สิ่งที่ต้องมี

ทั้งสองข้อต้องรันบน **Linux** (ทดสอบบน Ubuntu 24.04 ใน WSL2 บน Windows 11) เพราะต้องใช้ OpenMP และ `/proc`

```sh
sudo apt install build-essential python3            # gcc (มี OpenMP), make, python3
sudo apt install python3-matplotlib python3-numpy   # ใช้เฉพาะ plot.py
```

## ข้อ 1: Parallel matrix multiplication (OpenMP)

```sh
cd 1_parallel_6610505420
make                                    # ได้ไฟล์ ./parallel
./parallel -n 800 -t 8                  # n = 800, 8 threads, อัลกอริทึม blocked (ค่าเริ่มต้น)
./parallel -n 1600 -t 4 -a ikj -v       # เลือกอัลกอริทึม และแสดงว่า thread อยู่บน CPU ไหน
./parallel -h                           # ดูตัวเลือกทั้งหมด
```

| ตัวเลือก | ความหมาย |
|---|---|
| `-n N` | ขนาดเมทริกซ์ (1..20000) |
| `-t T` | จำนวน thread (ค่าเริ่มต้น = จำนวน CPU) |
| `-a ALGO` | `naive`, `ikj`, `transpose`, `blocked` |
| `-b BS` | ขนาด tile ของ `blocked` (ค่าเริ่มต้น 64) |
| `-s KIND -c CHUNK` | OpenMP schedule: `static` / `dynamic` / `guided` และ chunk size |
| `-r R` | คูณซ้ำ R รอบ แล้วรายงานเวลาที่ดีที่สุดและค่าเฉลี่ย |
| `-p` | กำหนดค่าเริ่มต้นของ A, B แบบขนาน (ค่าเริ่มต้นทำแบบ serial) |
| `--thp on/off` | ขอใช้หรือไม่ใช้ transparent huge pages ผ่าน `madvise()` |
| `--verify none/freivalds/full` | ตรวจผลลัพธ์ (ค่าเริ่มต้น: Freivalds' algorithm) |
| `--csv` | พิมพ์ผลเป็น CSV หนึ่งบรรทัด |

ค่าตั้งของ OpenMP runtime ส่งผ่าน environment variable ได้ เช่น
`OMP_PROC_BIND=close OMP_PLACES=cores ./parallel -n 1600 -t 8`

**รันการทดลองทั้งหมดและวาดกราฟ:**

```sh
make bench        # = python3 bench.py --out results   (ประมาณ 25-40 นาที)
make plot         # = python3 plot.py --data results  -> results/plots/*.png, results/summary.md
python3 bench.py --quick --out results   # ทดสอบแบบเร็ว (1-2 นาที)
```

## ข้อ 2: Memory map visualizer

`mmap.py` เป็น Python 3 ไฟล์เดียว ใช้แค่ standard library จึงไม่ต้องคอมไพล์ ส่วน `make` ใช้คอมไพล์โปรแกรมตัวอย่าง `demo`

```sh
cd 2_mmap_6610505420
make                                         # ได้ไฟล์ ./demo
python3 mmap.py --list                       # ดู PID ของโปรเซสที่ตรวจดูได้
python3 mmap.py self                         # snapshot ของโปรเซส python เอง
python3 mmap.py 1234                         # snapshot ของ PID 1234 (แสดงใน terminal)
python3 mmap.py 1234 --compact               # รวม region ของไฟล์/ไลบรารีเดียวกันเป็นแถวเดียว
python3 mmap.py 1234 --watch 1               # แบบ real time อัปเดตทุก 1 วินาที (Ctrl-C เพื่อออก)
python3 mmap.py 1234 --html map.html         # รายงาน HTML แบบโต้ตอบได้ (เปิดด้วย web browser)
python3 mmap.py 1234 --pages 12              # แสดงทีละหน้า (page) ว่าหน้าไหนของ region #12 อยู่ใน RAM
make run                                     # = python3 mmap.py --run ./demo --watch 0.5
python3 mmap.py --run ./demo --record 40 --html demo.html   # บันทึก timeline 40 วินาทีเป็น HTML
python3 mmap.py --file saved_smaps.txt       # วาดจากไฟล์ maps/smaps ที่บันทึกไว้
```

ข้อมูลขาเข้าคือ `/proc/<pid>/maps`, `/proc/<pid>/smaps`, `/proc/<pid>/pagemap`, `/proc/<pid>/status`
ตรวจดูได้เฉพาะโปรเซสของผู้ใช้คนเดียวกัน (หรือใช้ root)

## ความปลอดภัย (Do no harm)

- `parallel` ไม่ยอมรันถ้าต้องใช้หน่วยความจำเกิน 50% ของ RAM และจำกัดจำนวน thread ไม่เกิน 256
- การทดลองเรื่องการตั้งค่าระบบใช้เฉพาะสิ่งที่มีผลกับโปรเซสนั้นเท่านั้น (environment variable, `taskset`, `madvise`) ไม่มีการแก้ไขค่าของระบบ
- `bench.py` รันทีละโปรเซส
- `mmap.py` อ่านไฟล์ใน `/proc` อย่างเดียว ไม่เขียนหรือส่ง signal ไปยังโปรเซสอื่น ยกเว้นโปรเซสลูกที่ตัวเองเปิดด้วย `--run` ซึ่งจะถูกหยุดเมื่อจบการทำงาน
- `demo` ใช้ RAM ไม่เกินประมาณ 130 MiB และลบไฟล์ชั่วคราวทันทีที่สร้าง

## การใช้ AI

โค้ดทั้งหมดในงานนี้ (`parallel.c`, `bench.py`, `plot.py`, `mmap.py`, `demo.c`, Makefile) เขียนโดยใช้ AI coding assistant
**Claude Code (Anthropic, model Claude Opus 5.5)** เป็นผู้เขียน และ AI ได้รันการทดลอง (`bench.py`, `plot.py`) บนเครื่องของนิสิตด้วย
นิสิตเป็นผู้ตรวจสอบโค้ดและผลลัพธ์ และเขียนรายงานเอง
ระหว่างเขียนกราฟใน `plot.py` และหน้า HTML ของ `mmap.py` AI ได้โหลด skill ชื่อ `dataviz` ซึ่งรวมไว้ใน Claude Code (สำเนาอยู่ใน `SKILL.md` ของแต่ละโฟลเดอร์)
รายละเอียดการใช้ AI อยู่ในรายงานด้วย
