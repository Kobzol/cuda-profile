import React, {PureComponent} from 'react';
import {AccessType, Warp} from '../../../../lib/profile/warp';
import {MemoryAccess} from '../../../../lib/profile/memory-access';
import {getAccessesAddressRange, checkIntersectionRange} from '../../../../lib/profile/address';
import {AddressRange} from '../../../../lib/trace/selection';
import {formatDim3} from '../../../../lib/util/format';

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
                    fill={this.getAccessColor(warp, access, this.state.hovered)}
                    stroke='rgb(0, 0, 0)'
                    strokeWidth={0.5} />
                <title>{label}</title>
            </g>
        );
    }

    private handleMouseEnter = (event: React.MouseEvent<SVGSVGElement>) =>
    {
        if (this.props.selectionEnabled && this.props.access !== null)
        {
            this.props.onSelectChanged(getAccessesAddressRange([this.props.access], this.props.warp.size));
            this.setState(() => ({
                hovered: true
            }));
        }
    }
    private handleMouseLeave = (event: React.MouseEvent<SVGSVGElement>) =>
    {
        if (this.props.access !== null)
        {
            this.props.onSelectChanged(null);
            this.setState(() => ({
                hovered: false
            }));
        }
    }

    private getAccessColor = (warp: Warp, access: MemoryAccess, hovered: boolean): string =>
    {
        if (access === null) return 'rgb(255, 255, 255)';

        if (hovered)
        {
            return 'rgb(255, 0, 0)';
        }
        else if (this.props.memorySelection !== null &&
            checkIntersectionRange(this.props.memorySelection, access.address, warp.size))
        {
            return 'rgb(0, 255, 0)';
        }

        if (warp.accessType === AccessType.Read)
        {
            return 'rgb(225, 105, 65)';
        }
        else return 'rgb(65, 105, 225)';
    }
}
