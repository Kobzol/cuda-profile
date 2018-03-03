import React from 'react';
import {PureComponent} from 'react';
import {MemoryAllocation} from '../../../../lib/profile/memory-allocation';
import {MemoryRegion} from './memory-region';

interface Props
{
    allocations: MemoryAllocation[];
    height: number;
}

export class MemoryMap extends PureComponent<Props>
{
    render()
    {
        return (
            <svg
                width={'100%'}
                height={this.props.height}
                viewBox={'0 0 1000 ' + this.props.height}
                overflow={'visible'}>
                <rect
                    width={'100%'}
                    height={'100%'}
                    fill={'#000000'}
                    fillOpacity={'0.6'} />
                {this.renderAllocations(this.props.allocations)}
            </svg>
        );
    }

    renderAllocations = (allocations: MemoryAllocation[]): JSX.Element[] =>
    {
        const byteWidth = 4 * 1024;
        const rowHeight = 5;
        const padding = 10;
        let y = 0;

        return allocations.map(allocation => {
            const rows = this.getRows(allocation, byteWidth);
            let additionalRegion = 0;

            const leftover = allocation.size - (rows * byteWidth);
            if (leftover > 0)
            {
                additionalRegion = leftover / byteWidth;
            }

            const regionHeight = rows * rowHeight;
            const loc = y;
            y += regionHeight + padding;

            return (
                <MemoryRegion
                    key={allocation.address}
                    allocation={allocation}
                    byteWidth={byteWidth}
                    height={regionHeight}
                    rowHeight={rowHeight}
                    y={loc}
                    additionalRegion={additionalRegion * 100.0}
                />
            );
        });
    }

    getRows = (allocation: MemoryAllocation, byteWidth: number) =>
    {
        return Math.floor(allocation.size / byteWidth);
    }
}
