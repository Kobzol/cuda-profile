import React, {PureComponent} from 'react';
import {AccessType, Warp} from '../../../../lib/profile/warp';
import {MemoryAccess} from '../../../../lib/profile/memory-access';
import {getAccessesAddressRange, checkIntersectionRange} from '../../../../lib/profile/address';
import {AddressRange} from '../../../../lib/trace/selection';
import {formatDim3} from '../../../../lib/util/format';
import _ from 'lodash';
import {Color} from 'chroma-js';
import * as chroma from 'chroma-js';

interface Props
{
    x: number;
    y: number;
    width: number;
    height: number;
    warp: Warp;
    access: MemoryAccess;
    memorySelection: AddressRange[];
    onSelectChanged: (range: AddressRange | null) => void;
    selectionEnabled: boolean;
}

interface State
{
    hovered: boolean;
}

export class Thread extends PureComponent<Props, State>
{
    constructor(props: Props)
    {
        super(props);

        this.state = {
            hovered: false
        };
    }

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
                onMouseLeave={this.handleMouseLeave}>
                <rect
                    x={this.props.x}
                    y={this.props.y}
                    width={this.props.width}
                    height={this.props.height}
                    fill={this.getAccessColor(warp, access, this.state.hovered).hex()}
                    stroke='rgb(40, 40, 40)'
                    strokeWidth={this.state.hovered ? 0.75 : 0.35} />
                <title>{label}</title>
            </g>
        );
    }

    handleMouseEnter = () =>
    {
        if (this.props.selectionEnabled && this.props.access !== null)
        {
            this.props.onSelectChanged(getAccessesAddressRange([this.props.access], this.props.warp.size));
        }
        this.setState(() => ({
            hovered: true
        }));
    }
    handleMouseLeave = () =>
    {
        if (this.props.access !== null)
        {
            this.props.onSelectChanged(null);
        }
        this.setState(() => ({
            hovered: false
        }));
    }

    getColorForAccessType = (warp: Warp, access: MemoryAccess): Color =>
    {
        if (access === null) return chroma(240, 240, 240);
        if (warp.accessType === AccessType.Read) return chroma(20, 180, 20);
        return chroma(180, 20, 0);
    }

    getAccessColor = (warp: Warp, access: MemoryAccess, hovered: boolean): Color =>
    {
        const color = this.getColorForAccessType(warp, access);

        if (hovered)
        {
            return color.darken(1.25);
        }
        else if (this.props.memorySelection !== null &&
                access !== null &&
            _.some(this.props.memorySelection, selection =>
                checkIntersectionRange(selection, access.address, warp.size)))
        {
            return color.darken(1.15);
        }

        return color;
    }
}
