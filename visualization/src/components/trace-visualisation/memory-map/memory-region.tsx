import * as React from 'react';
import {PureComponent} from 'react';
import {MemoryAllocation} from '../../../lib/profile/memory-allocation';
import {formatByteSize} from '../../../lib/util/format';
import {AddressSpace} from '../../../lib/profile/warp';

interface Props
{
    allocation: MemoryAllocation;
    byteWidth: number;
    height: number;
    rowHeight: number;
    y: number;
    additionalRegion: number;
}

export class MemoryRegion extends PureComponent<Props>
{
    render()
    {
        const fill = this.getColor(this.props.allocation);
        const {address, size, elementSize, type} = this.props.allocation;
        const count = size / elementSize;

        return (
            <g>
                <title>{address}: {formatByteSize(size)} MiB ({count}x {type})</title>
                <g y={this.props.y} transform='translate(-50, 0)'>
                    <text
                        y={this.props.y + this.props.rowHeight}
                        textAnchor={'middle'}
                        dominantBaseline={'central'}>{address}</text>
                </g>
                <rect
                    y={this.props.y}
                    width={'100%'}
                    height={this.props.height + 'px'}
                    fill={fill} />
                {this.props.additionalRegion !== 0 &&
                    <rect
                        y={this.props.y + this.props.height}
                        width={this.props.additionalRegion + '%'}
                        height={this.props.rowHeight}
                        fill={fill} />
                }
            </g>
        );
    }

    getColor = (allocation: MemoryAllocation): string =>
    {
        switch (allocation.space)
        {
            case AddressSpace.Constant: return '#0000FF';
            case AddressSpace.Shared: return '#00FF00';
            case AddressSpace.Global: return '#FF0000';
            default: return '#000000';
        }
    }
}
