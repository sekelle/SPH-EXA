/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Neighbor list compression
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

// if 1, use compression proposed in Compressed Neighbour Lists for SPH, by S. Band, C. Gissler and M. Teschner, 2020
#ifndef CSTONE_USE_BAND_ET_AL_COMPRESSION
#define CSTONE_USE_BAND_ET_AL_COMPRESSION 0
#endif

#include <cassert>
#include <cstdint>

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/clz.hpp"
#include "cstone/primitives/warpscan.cuh"

namespace cstone
{

/*! compress a list of neighbor indices with a single warp
 *
 * This function compresses an array of neighbor indices using either the compression scheme proposed in 'Compressed
 * Neighbour Lists for SPH', by S. Band, C. Gissler and M. Teschner, 2020 or a custom nibble-based scheme, depending on
 * the CSTONE_USE_BAND_ET_AL_COMPRESSION macro.
 *
 * Note that the input values need to be the same for all threads in a warp. Caution: if the output buffer is too small,
 * it will overflow.
 *
 * @param[in]  neighbors  pointer to the array of neighbor indices to compress
 * @param[out] output     pointer to the output buffer where compressed data will be written
 * @param[in]  n          number of neighbor indices in the input array
 */
__device__ __forceinline__ void
warpCompressNeighbors(const std::uint32_t* __restrict__ neighbors, char* __restrict__ output, const unsigned n)
{
    const unsigned laneIdx = laneIndex();

    if (n == 0)
    {
        if (laneIdx == 0) *((unsigned*)output) = sizeof(unsigned);
        return;
    }

#if CSTONE_USE_BAND_ET_AL_COMPRESSION
    GpuConfig::ThreadMask* control = (GpuConfig::ThreadMask*)output + 1;
    std::uint32_t* first = (std::uint32_t*)(control + (n - 1 + GpuConfig::warpSize - 1) / GpuConfig::warpSize * 2);
    std::uint8_t* data   = (std::uint8_t*)(first + 1);

    unsigned dataSize = 0;
    unsigned previous = neighbors[0];
    if (laneIdx == 0) *first = previous;
    for (unsigned offset = 1; offset < n; offset += GpuConfig::warpSize)
    {
        const unsigned nb           = offset + laneIdx;
        const unsigned neighbor     = nb < n ? neighbors[nb] : 0;
        const unsigned leftNeighbor = shflUpSync(neighbor, 1);
        const unsigned diff         = nb < n ? (neighbor - (laneIdx > 0 ? leftNeighbor : previous)) - 1 : 0;
        previous                    = shflSync(neighbor, GpuConfig::warpSize - 1);

        const auto firstControl  = ballotSync(diff > 1);
        const auto secondControl = ballotSync((diff == 1) | (diff >= 256));
        if (laneIdx == 0)
        {
            control[2 * ((offset - 1) / GpuConfig::warpSize)]     = firstControl;
            control[2 * ((offset - 1) / GpuConfig::warpSize) + 1] = secondControl;
        }

        const unsigned dataBytes      = diff >= 2 ? (diff >= 256 ? 4 : 1) : 0;
        const unsigned dataBytesScan  = inclusiveScanInt(dataBytes);
        const unsigned dataBytesIndex = dataSize + dataBytesScan - dataBytes;
        dataSize += shflSync(dataBytesScan, GpuConfig::warpSize - 1);

        for (unsigned i = 0; i < dataBytes; ++i)
            data[dataBytesIndex + i] = (diff >> (8 * i)) & 0xff;
    }

    const unsigned totalBytes =
        sizeof(GpuConfig::ThreadMask) * (1 + (n - 1 + GpuConfig::warpSize - 1) / GpuConfig::warpSize * 2) + 4 +
        dataSize;
    assert(n < (1 << 16));
    if (laneIdx == 0) *((unsigned*)output) = totalBytes | (n << 16);
#else
    GpuConfig::ThreadMask* nonOnes = (GpuConfig::ThreadMask*)output + 1;
    std::uint8_t* data             = (std::uint8_t*)(nonOnes + (n + GpuConfig::warpSize - 1) / GpuConfig::warpSize);

    const auto writeDataNibble = [&](unsigned index, std::uint8_t value, bool odd)
    {
        assert(value < 16);
        if (odd == index % 2)
        {
            std::uint8_t byte = odd ? data[index / 2] : 0;
            byte |= (value << ((index % 2) * 4));
            data[index / 2] = byte;
        }
    };

    unsigned dataSize = 0;
    int previous      = -1;
    for (unsigned offset = 0; offset < n; offset += GpuConfig::warpSize)
    {
        const unsigned nb           = offset + laneIdx;
        const unsigned neighbor     = nb < n ? neighbors[nb] : 0;
        const unsigned leftNeighbor = shflUpSync(neighbor, 1);
        const unsigned diff         = neighbor - (laneIdx > 0 ? leftNeighbor : previous);
        previous                    = shflSync(neighbor, GpuConfig::warpSize - 1);

        const bool nonOne     = diff != 1 & nb < n;
        const auto nonOneBits = ballotSync(nonOne);
        if (laneIdx == 0) nonOnes[offset / GpuConfig::warpSize] = nonOneBits;
        const bool additionalStorage = (diff > 9) & (nb < n);
        const unsigned nBits         = 32 - countLeadingZeros(diff);
        const unsigned nNibbles      = additionalStorage ? (nBits + 3) / 4 : 0;
        const unsigned nNibblesData  = additionalStorage ? nNibbles - 1 : diff + 6;

        const unsigned nNibblesIndex     = exclusiveScanBool(nonOne);
        const unsigned nNibblesDataIndex = dataSize + nNibblesIndex;
        dataSize += popCount(nonOneBits);

        if (nonOne) writeDataNibble(nNibblesDataIndex, nNibblesData, false);
#ifdef __HIP_PLATFORM_AMD__
        // This should not be necessary, a memory fence should be enough, but tests fail without
        __syncthreads();
#else
        syncWarp();
#endif
        if (nonOne) writeDataNibble(nNibblesDataIndex, nNibblesData, true);

        const unsigned nbValueScan      = inclusiveScanInt(nNibbles);
        const unsigned nbValueDataIndex = dataSize + nbValueScan - nNibbles;
        const unsigned nbValueSize      = shflSync(nbValueScan, GpuConfig::warpSize - 1);
        dataSize += nbValueSize;

        for (unsigned i = 0; i < nNibbles; ++i)
            writeDataNibble(nbValueDataIndex + i, (diff >> (4 * i)) & 0xf, false);
#ifdef __HIP_PLATFORM_AMD__
        // This should not be necessary, a memory fence should be enough, but tests fail without
        __syncthreads();
#else
        syncWarp();
#endif
        for (unsigned i = 0; i < nNibbles; ++i)
            writeDataNibble(nbValueDataIndex + i, (diff >> (4 * i)) & 0xf, true);
    }

    const unsigned totalBytes =
        sizeof(GpuConfig::ThreadMask) * (1 + (n + GpuConfig::warpSize - 1) / GpuConfig::warpSize) + (dataSize + 1) / 2;
    assert(totalBytes < (1 << 16));
    assert(n < (1 << 16));
    if (laneIdx == 0) *((unsigned*)output) = totalBytes | (n << 16);
#endif
}

/*! extract the size of a neighbor list compressed by warpCompressNeighbors
 *
 * @param[in] input pointer to the buffer containing the compressed neighbor list
 * @return          the size (in bytes) of the compressed neighbor list
 */
__device__ __forceinline__ unsigned compressedNeighborsSize(const char* const input)
{
    return *((const unsigned*)input) & 0xffff;
}

/*! @brief Extract up to 32-bit from a bitstream represented as 32 bit integers per lane
 *
 * @param bitstream  a bitstream of length GpuConfig::warpSize x 32 bits = 1024/2048 bits, each lane holds 32 bits
 * @param firstBit   bit index in range [0:1024/2048] to start extraction
 * @param lastBit    bit index in range [0:1024/2048] to extract up to (exclusive)
 * @return           bitstream[firstBit:lastBit] as 32-bit integer
 */
__device__ __forceinline__ unsigned extractFromBitstream(unsigned bitstream, unsigned firstBit, unsigned lastBit)
{
    constexpr unsigned bitsPerLane = CHAR_BIT * sizeof(unsigned);
    assert(firstBit <= lastBit && lastBit <= GpuConfig::warpSize * bitsPerLane);

    unsigned lane1             = firstBit / bitsPerLane;
    unsigned data1             = shflSync(bitstream, lane1);
    unsigned lane1numValidBits = std::min(lastBit, bitsPerLane * (lane1 + 1)) - firstBit;
    unsigned lane1validMask    = (1u << lane1numValidBits) - 1;
    unsigned lane1contrib      = (data1 >> (firstBit % bitsPerLane)) & lane1validMask;

    unsigned lane2             = lastBit > 0 ? (lastBit - 1) / bitsPerLane : 0;
    unsigned data2             = shflSync(bitstream, lane2);
    unsigned lane2numValidBits = lane2 > lane1 ? lastBit - lane2 * bitsPerLane : 0;
    unsigned lane2validMask    = (1u << lane2numValidBits) - 1;
    unsigned lane2contrib      = (data2 & lane2validMask) << lane1numValidBits;

    return lane1contrib |= lane2contrib;
}

/*! @brief Load @p warpNumNb 4-bit nibbles starting from data + 4-bit * nbStartIdx into 32-bit integers
 *
 * @param data
 * @param nbStartIdx  offset in nibbles relative to @p data to start reading from
 * @param warpNumNb   total number of nibbles in warp to read from stream
 * @return            stream data read from memory and number of extra nibbles read at stream start for alignment
 */
__device__ __forceinline__ std::tuple<unsigned, unsigned>
loadBitStream(const void* data, unsigned nbStartIdx, unsigned warpNumNb)
{
    const unsigned laneIdx = laneIndex();
    const unsigned stream32BitStartIdx    = nbStartIdx / 8; // round down to 4-byte multiple
    const unsigned streamNbOffset         = nbStartIdx % 8;
    const unsigned streamNum32BitSegments = (warpNumNb + streamNbOffset + 7) / 8; // round up to multiples of 8

    unsigned streamData = 0;
    if (laneIdx < streamNum32BitSegments)
        streamData = reinterpret_cast<const unsigned*>(data)[stream32BitStartIdx + laneIdx];

    return {streamData, streamNbOffset};
}

/*! decompress a list of neighbor indices which was compressed using warpCompressNeighbors with a single warp
 *
 * The function reads the compressed neighbor list from the input buffer and reconstructs
 * the original neighbor indices, storing them in the provided neighbors array.
 * The number of decompressed neighbor indices is returned via the reference parameter n.
 *
 * @param[in]  input     pointer to the buffer containing the compressed neighbor list
 * @param[out] neighbors pointer to the array where decompressed neighbor indices will be stored
 * @param[out] n         reference to an unsigned integer where the number of neighbor indices will be stored
 */
__device__ __forceinline__ void
warpDecompressNeighbors(const char* const __restrict__ input, std::uint32_t* const __restrict__ neighbors, unsigned& n)
{
    const unsigned laneIdx = laneIndex();

    n = *((unsigned*)input) >> 16;

    if (n == 0) return;

#if CSTONE_USE_BAND_ET_AL_COMPRESSION
    const GpuConfig::ThreadMask* control = (const GpuConfig::ThreadMask*)input + 1;
    const std::uint32_t* first =
        (std::uint32_t*)(control + (n - 1 + GpuConfig::warpSize - 1) / GpuConfig::warpSize * 2);
    const std::uint8_t* data = (std::uint8_t*)(first + 1);

    unsigned dataSize = 0;
    unsigned previous = *first;
    if (laneIdx == 0) neighbors[0] = previous;
    for (unsigned offset = 1; offset < n; offset += GpuConfig::warpSize)
    {
        const unsigned nb        = offset + laneIdx;
        const auto firstControl  = control[2 * ((offset - 1) / GpuConfig::warpSize)];
        const auto secondControl = control[2 * ((offset - 1) / GpuConfig::warpSize) + 1];

        const bool firstControlBit  = (firstControl >> laneIdx) & 1;
        const bool secondControlBit = (secondControl >> laneIdx) & 1;

        unsigned diff            = !firstControlBit & secondControlBit;
        const unsigned dataBytes = firstControlBit ? (secondControlBit ? 4 : 1) : 0;

        const unsigned dataBytesScan  = inclusiveScanInt(dataBytes);
        const unsigned dataBytesIndex = dataSize + dataBytesScan - dataBytes;
        dataSize += shflSync(dataBytesScan, GpuConfig::warpSize - 1);

        previous = shflSync(previous, GpuConfig::warpSize - 1);

        for (unsigned i = 0; i < dataBytes; ++i)
            diff |= unsigned(data[dataBytesIndex + i]) << (8 * i);

        previous += inclusiveScanInt(diff + 1);

        if (nb < n) neighbors[nb] = previous;
    }
#else
    const GpuConfig::ThreadMask* nonOnes = (const GpuConfig::ThreadMask*)input + 1;
    const std::uint8_t* data = (const std::uint8_t*)(nonOnes + (n + GpuConfig::warpSize - 1) / GpuConfig::warpSize);

    const auto readDataNibble = [data](unsigned index)
    {
        const unsigned byte = data[index / 2];
        return (byte >> ((index % 2) * 4)) & 0xf;
    };

    unsigned dataSize = 0;
    int previous      = -1;
    for (unsigned offset = 0; offset < n; offset += GpuConfig::warpSize)
    {
        const auto nonOneBits = nonOnes[offset / GpuConfig::warpSize];
        const bool nonOne     = (nonOneBits >> laneIdx) & 1;

        // nibble info section -> number of data nibbles or immediate value, max = warpSize nibbles (16 bytes)
        const unsigned nNibbleIndex = dataSize + popCount(nonOneBits & lanemask_lt());
        dataSize += popCount(nonOneBits);

        // read nibble info
        const unsigned nNibblesData  = nonOne ? readDataNibble(nNibbleIndex) : 0; // info nibble value
        const bool additionalStorage = nonOne ? nNibblesData <= 7 : 0;
        const unsigned nNibbles      = additionalStorage ? nNibblesData + 1 : 0; // max nNibbles is 8

        // nibble data section
        const unsigned nbValueScan      = inclusiveScanInt(nNibbles);
        const unsigned nbValueDataIndex = nbValueScan - nNibbles;                         // convert to exclusive scan
        const unsigned nbValueSize      = shflSync(nbValueScan, GpuConfig::warpSize - 1); // warpSum of nNibbles

        previous = shflSync(previous, GpuConfig::warpSize - 1);

        const auto [streamData, nbAlign] = loadBitStream(data, dataSize, nbValueSize);
        const auto streamBitStart        = 4 * (nbValueDataIndex + nbAlign);
        const auto streamBitEnd          = 4 * (nbValueDataIndex + nbAlign + nNibbles);
        const auto streamExtract         = extractFromBitstream(streamData, streamBitStart, streamBitEnd);
        unsigned diff                    = nonOne ? (additionalStorage ? streamExtract : nNibblesData - 6) : 1;
        dataSize += nbValueSize;

        previous += inclusiveScanInt(diff);
        const unsigned nb = offset + laneIdx;
        if (nb < n) neighbors[nb] = previous;
    }
#endif
}

} // namespace cstone
