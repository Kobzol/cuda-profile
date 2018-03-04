import React, {PureComponent} from 'react';
import {AccessType, Warp} from '../../../../lib/profile/warp';
import {MemoryAccess} from '../../../../lib/profile/memory-access';
import {checkIntersectionRange} from '../../../../lib/profile/address';
import {AddressRange, createWarpAccess, WarpAccess} from '../../../../lib/trace/selection';
import {formatDim3} from '../../../../lib/util/format';
import {Color, default as chroma} from 'chroma-js';
import {any} from 'ramda';
import {getIdentifier, READ_COLOR, WRITE_COLOR} from '../../warp-access-settings';

interface Props
{
    x: number;
    y: number;
    width: number;
    height: number;
    warp: Warp;
    access: MemoryAccess;
    memorySelection: AddressRange[];
    selected: boolean;
    selectedIndex: number;
    onSelectChanged(range: WarpAccess, active: boolean): void;
}

interface State
{
    hovered: boolean;
    clicked: boolean;
}

export class Thread extends PureComponent<Props, State>
{
    state: State = {
        hovered: false,
        clicked: false
    };

    render()
    {
        const {access, warp} = this.props;

        let label = 'Inactive thread';
        if (access !== null)
        {
            const dim = formatDim3(access.threadIdx);
            label = `${dim}: ${access.address}`;
        }

        return (
            <g
                onMouseEnter={this.handleMouseEnter}
                onMouseLeave={this.handleMouseLeave}
                onClick={this.handleClick}>
                <rect
                    x={this.props.x}
                    y={this.props.y}
                    width={this.props.width}
                    height={this.props.height}
                    fill={this.getAccessColor(warp, access).hex()}
                    stroke='rgb(40, 40, 40)'
                    strokeWidth={this.state.hovered ? 0.75 : 0.35} />
                {this.props.selected &&
                    <text
                        x={this.props.x + this.props.width / 2}
                        y={this.props.y + this.props.height / 2}
                        textAnchor='middle'
                        alignmentBaseline='central'
                        fill='rgb(255, 255, 255)'
                        fontSize='14px'>
                        {getIdentifier(this.props.selectedIndex)}
                    </text>
                }
                <title>{label}</title>
            </g>
        );
    }

    handleClick = () =>
    {
        if (this.state.clicked)
        {
            this.deselect();
        }
        else this.select();

        this.setState(state => ({
            clicked: !state.clicked
        }));
    }
    handleMouseEnter = () =>
    {
        this.select();
        this.setState(() => ({
            hovered: true
        }));
    }
    handleMouseLeave = () =>
    {
        this.setState(() => ({
            hovered: false
        }));
        if (!this.state.clicked)
        {
            this.deselect();
        }
    }

    select = () =>
    {
        if (this.props.access !== null && !this.props.selected)
        {
            this.props.onSelectChanged(createWarpAccess(this.props.warp, this.props.access), true);
        }
    }
    deselect = () =>
    {
        if (this.props.access !== null && this.props.selected)
        {
            this.props.onSelectChanged(createWarpAccess(this.props.warp, this.props.access), false);
        }
    }

    getColorForAccessType = (warp: Warp, access: MemoryAccess): Color =>
    {
        if (access === null) return chroma(240, 240, 240);
        if (warp.accessType === AccessType.Read) return READ_COLOR;
        return WRITE_COLOR;
    }
    getAccessColor = (warp: Warp, access: MemoryAccess): Color =>
    {
        const color = this.getColorForAccessType(warp, access);

        if (this.props.selected)
        {
            return color.darken(1.25);
        }
        else if (this.props.memorySelection !== null &&
                access !== null &&
                any(selection => checkIntersectionRange(selection, access.address, warp.size),
                    this.props.memorySelection))
        {
            return color.darken(1.15);
        }

        return color;
    }
}
