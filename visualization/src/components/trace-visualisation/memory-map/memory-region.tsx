import * as React from 'react';
import {PureComponent} from 'react';
import {MemoryAllocation} from '../../../lib/format/memory-allocation';
import {formatMiB} from '../../../lib/util/format';

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
        const fill = '#0000FF';
        const {address, size, elementSize, typeString} = this.props.allocation;
        const count = size / elementSize;

        return (
            <g>
                <title>{address}: {formatMiB(size)} MiB ({count}x {typeString})</title>
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
}
