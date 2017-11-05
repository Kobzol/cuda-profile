import React, {PureComponent} from 'react';
import {AccessType, Warp} from '../../../../lib/profile/warp';
import {MemoryAccess} from '../../../../lib/profile/memory-access';
import {getAccessAddressRange, checkIntersectionRange} from '../../../../lib/profile/address';
import {AddressRange} from '../../../../lib/trace/selection';

interface Props
{
    x: number;
    y: number;
    width: number;
    height: number;
    warp: Warp;
    access: MemoryAccess;
    memorySelection: AddressRange;
    onSelectChanged: (range: AddressRange | null) => void;
}

export class Thread extends PureComponent<Props>
{
    render()
    {
        const {access, warp} = this.props;

        let label = 'Inactive thread';
        if (access !== null)
        {
            const {x: tx, y: ty, z: tz} = access.threadIdx;
            label = `${warp.blockIdx.z}.${warp.blockIdx.y}.${warp.blockIdx.x}.` +
                    `${tz}.${ty}.${tx}: ${warp.size} at ${access.address}`;
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
                    fill={this.getAccessColor(warp, access)}
                    stroke='rgb(0, 0, 0)'
                    strokeWidth={0.5} />
                <title>{label}</title>
            </g>
        );
    }

    private handleMouseEnter = (event: React.MouseEvent<SVGSVGElement>) =>
    {
        if (this.props.access !== null)
        {
            this.props.onSelectChanged(getAccessAddressRange([this.props.access], this.props.warp.size));
        }
    }
    private handleMouseLeave = (event: React.MouseEvent<SVGSVGElement>) =>
    {
        if (this.props.access !== null)
        {
            this.props.onSelectChanged(null);
        }
    }

    private getAccessColor = (warp: Warp, access: MemoryAccess): string =>
    {
        if (access === null) return 'rgb(255, 255, 255)';

        if (this.props.memorySelection !== null)
        {
            if (checkIntersectionRange(this.props.memorySelection, access.address, warp.size))
            {
                return 'rgb(0, 255, 0)';
            }
        }
        if (warp.kind === AccessType.Read)
        {
            return 'rgb(255, 0, 0)';
        }
        else return 'rgb(0, 0, 255)';
    }
}
