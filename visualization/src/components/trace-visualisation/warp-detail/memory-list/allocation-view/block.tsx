import React, {PureComponent} from 'react';
import {AddressRange, WarpAccess} from '../../../../../lib/trace/selection';
import {Color, default as chroma} from 'chroma-js';
import {AccessType} from '../../../../../lib/profile/warp';
import {getIdentifier, READ_COLOR, WRITE_COLOR} from '../../../warp-access-settings';

interface Props
{
    x: number;
    y: number;
    width: number;
    height: number;
    range: AddressRange;
    access: WarpAccess | null;
    accessIndex: number;
    onMemorySelect(range: AddressRange | null): void;
}

export class Block extends PureComponent<Props>
{
    render()
    {
        return (
            <g onMouseEnter={this.handleMouseEnter}
               onMouseLeave={this.handleMouseLeave}>
                <rect
                    x={this.props.x}
                    y={this.props.y}
                    width={this.props.width}
                    height={this.props.height}
                    fill={this.getFillColor().hex()}
                    stroke='rgb(40, 40, 40)'
                    strokeWidth={0.35} />
                {this.props.access !== null &&
                    <text
                        x={this.props.x + this.props.width / 2}
                        y={this.props.y + this.props.height / 2}
                        textAnchor='middle'
                        alignmentBaseline='central'
                        fill='rgb(255, 255, 255)'
                        fontSize='14px'>
                        {getIdentifier(this.props.accessIndex)}
                    </text>
                }
                <title>{this.props.range.from} - {this.props.range.to}</title>
            </g>
        );
    }

    handleMouseEnter = () =>
    {
        this.props.onMemorySelect(this.props.range);
    }
    handleMouseLeave = () =>
    {
        this.props.onMemorySelect(null);
    }

    getFillColor = (): Color =>
    {
        if (this.props.access !== null)
        {
            if (this.props.access.warp.accessType === AccessType.Read)
            {
                return READ_COLOR;
            }
            else return WRITE_COLOR;
        }
        return chroma(255, 255, 255);
    }
}
