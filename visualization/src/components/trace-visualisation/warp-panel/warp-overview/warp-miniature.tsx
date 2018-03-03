import React, {PureComponent} from 'react';
import {Warp} from '../../../../lib/profile/warp';
import {formatDim3} from '../../../../lib/util/format';
import {Color} from 'chroma-js';
import chroma from 'chroma-js';

interface Props
{
    x: number;
    y: number;
    width: number;
    height: number;
    warp: Warp;
    selected: boolean;
    onClick(warp: Warp, ctrlPressed: boolean): void;
}

interface State
{
    hovered: boolean;
}

export class WarpMiniature extends PureComponent<Props, State>
{
    state: State = {
        hovered: false
    };

    render()
    {
        const {warp} = this.props;
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
                    fill={this.getFillColor(warp, this.state.hovered).hex()}
                    stroke='rgb(40, 40, 40)'
                    strokeWidth={this.state.hovered ? 0.75 : 0.35} />
                <title>{`${formatDim3(warp.blockIdx)}`}</title>
            </g>
        );
    }

    handleMouseEnter = () =>
    {
        this.setState(() => ({
            hovered: true
        }));
    }
    handleMouseLeave = () =>
    {
        this.setState(() => ({
            hovered: false
        }));
    }
    handleClick = (e: React.MouseEvent<SVGGElement>) =>
    {
        this.props.onClick(this.props.warp, e.ctrlKey);
    }

    getFillColor = (warp: Warp, hovered: boolean): Color =>
    {
        return this.props.selected ? chroma(140, 0, 140) : chroma(240, 240, 240);
    }
}
