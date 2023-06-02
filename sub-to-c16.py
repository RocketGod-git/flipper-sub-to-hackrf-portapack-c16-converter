import os
import argparse
import math
from typing import List, Tuple
import struct
import numpy as np

def str2abbr(str_: str = '') -> str:
    return ''.join(word[0] for word in str_.split('_'))

SUPPORTED_PROTOCOLS = ['RAW']

def parse_Sub(file: str) -> dict:
    try:
        with open(file, 'r') as f:
          sub_data = f.read()
    except:
        print('Cannot read input file')
        exit(-1)

    sub_chunks = [r.strip() for r in sub_data.split('\n')]
    info = {k.lower(): v.strip() for k, v in (row.split(':') for row in sub_chunks[:5])}

    print(f'Read info from file: {info}')

    if info.get('protocol') not in SUPPORTED_PROTOCOLS:
        print(f'Failed to parse {file}: Currently supported protocols are {", ".join(SUPPORTED_PROTOCOLS)} (found: {info.get("protocol")})')
        exit(-1)

    info['chunks'] = [
        list(map(int, r.split(':')[1].split()))
        for r in sub_chunks[5:]
        if ':' in r
    ]

    return info

def write_HRF_file(file: str, buffer: bytes, frequency: str, sampling_rate: str) -> List[str]:
    PATHS = [f'{file}.{ext}' for ext in ['C16', 'TXT']]
    with open(PATHS[0], 'wb') as f:
        for chunk in buffer:
            f.write(bytes(chunk))
    with open(PATHS[1], 'w') as f:
        f.write(generate_meta_string(frequency, sampling_rate))
    return PATHS

def generate_meta_string(frequency: str, sampling_rate: str) -> str:
    meta = [['sample_rate', sampling_rate], ['center_frequency', frequency]]
    return '\n'.join('='.join(map(str, r)) for r in meta)

HACKRF_OFFSET = 0

import numpy as np

def durations_to_bin_sequence(durations: List[List[int]], sampling_rate: int, intermediate_freq: int, amplitude: int) -> List[Tuple[int, int]]:
    sequence = []
    for chunk in durations:
        for duration in chunk:
            sequence.extend(us_to_sin(duration > 0, abs(duration), sampling_rate, intermediate_freq, amplitude))
    return sequence

def us_to_sin(level: bool, duration: int, sampling_rate: int, intermediate_freq: int, amplitude: int) -> List[Tuple[int, int]]:
    ITERATIONS = int(sampling_rate * duration / 1_000_000)
    if ITERATIONS == 0:
        return []

    DATA_STEP_PER_SAMPLE = 2 * math.pi * intermediate_freq / sampling_rate

    HACKRF_AMPLITUDE = (256 ** 2 - 1) * (amplitude / 100)

    return [
        (
            HACKRF_OFFSET + int(math.floor(math.cos(i * DATA_STEP_PER_SAMPLE) * (HACKRF_AMPLITUDE / 2))),
            HACKRF_OFFSET + int(math.floor(math.sin(i * DATA_STEP_PER_SAMPLE) * (HACKRF_AMPLITUDE / 2)))
        )
        if level else (HACKRF_OFFSET, HACKRF_OFFSET)
        for i in range(ITERATIONS)
    ]

def sequence_to_16LEBuffer(sequence: List[Tuple[int, int]]) -> bytes:
    return np.array(sequence).astype(np.int16).tobytes()

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="SDR-based file processing script")
    parser.add_argument('file', help="Input file path.")
    parser.add_argument('-o', '--output', help="Output file path. If not specified, the input file name will be used.")
    parser.add_argument('-sr', '--sampling_rate', type=int, default=500000, help="Sampling rate for the output file. Default is 500ks/s.")
    parser.add_argument('-if', '--intermediate_freq', type=int, default=None, help="Intermediate frequency.")
    parser.add_argument('-a', '--amplitude', type=int, default=100, help="Amplitude percentage. Default is 100.")
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output.')
    return vars(parser.parse_args())

args = parse_args()

file = args.get('file')
output = args.get('output', os.path.splitext(file)[0])
sampling_rate = args.get('sampling_rate', 500000)
intermediate_freq = args.get('intermediate_freq') or sampling_rate // 100
amplitude = args.get('amplitude', 100)

info = parse_Sub(file)
print(f'Sub File information: {info}')

chunks = info.get('chunks', [])
print(f'Found {len(chunks)} pure data chunks')

IQSequence = durations_to_bin_sequence(chunks, sampling_rate, intermediate_freq, amplitude)
buff = sequence_to_16LEBuffer(IQSequence)
outFiles = write_HRF_file(output, buff, info['frequency'], sampling_rate)
print(f'Written {round(len(buff) / 1024)} kiB, {len(IQSequence) / sampling_rate} seconds in files {", ".join(outFiles)}')
