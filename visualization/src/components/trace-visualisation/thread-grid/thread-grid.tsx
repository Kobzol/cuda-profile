import React, {PureComponent} from 'react';
import {MemoryAccess, Warp} from '../../../lib/profile/memory-access';
import {Trace} from '../../../lib/profile/trace';
import {getLaneId, getWarpStart} from '../../../lib/profile/api';
import {createBlockSelector} from './grid/grid-data';
import {Selector} from 'reselect';
import {Dictionary} from 'lodash';
import GridLayout from 'd3-v4-grid';
import * as _ from 'lodash';

interface Props
{
    trace: Trace;
    warp: Warp;
    canvasDimensions: { width: number, height: number };
}
interface State
{
    blockMapSelector: Selector<Warp, Dictionary<MemoryAccess>>;
}

export class ThreadGrid extends PureComponent<Props, State>
{
    constructor(props: Props)
    {
        super(props);

        this.state = {
            blockMapSelector: createBlockSelector()
        };
    }

    render()
    {
        const layout = this.calculateLayout(
            { rows: 4, cols: 8 },
            { width: this.props.canvasDimensions.width, height: this.props.canvasDimensions.height }
        );
        const nodeSize = {
            width: layout.nodeSize()[0],
            height: layout.nodeSize()[1]
        };

        const grid = this.renderGrid(layout.nodes(), nodeSize);

        return (
            <svg width='100%' height='100%'
                 viewBox={`0 0 ${this.props.canvasDimensions.width} ${this.props.canvasDimensions.height}`}>
                <g>{grid}</g>
            </svg>
        );
    }

    renderGrid = (nodes: Array<{x: number, y: number}>,
                  nodeSize: {width: number, height: number}): JSX.Element[] =>
    {
        const grid: JSX.Element[] = [];
        const width = 8;
        const height = 4;
        
        const accesses = this.createWarpAccesses(this.props.trace, this.props.warp);

        for (let y = 0; y < height; y++)
        {
            for (let x = 0; x < width; x++)
            {
                const index = y * (width) + x;
                const access = accesses[index];

                grid.push(
                    <rect
                        key={index}
                        x={nodes[index].x}
                        y={nodes[index].y}
                        width={nodeSize.width}
                        height={nodeSize.height}
                        fill={access === null ? 'rgb(255, 255, 255)' : 'rgb(255, 0, 0)'}
                        stroke='rgb(0, 0, 0)'
                        strokeWidth={1} />
                );
            }
        }

        return grid;
    }

    calculateLayout(gridSize: {rows: number, cols: number},
                    canvasSize: {width: number, height: number})
    {
        const layout = GridLayout()
            .data(_.range(gridSize.rows * gridSize.cols))
            .padding([0.1, 0.1])
            .bands(true)
            .rows(gridSize.rows)
            .cols(gridSize.cols)
            .size([canvasSize.width, canvasSize.height]);
        layout.layout();

        return layout;
    }

    private createWarpAccesses(trace: Trace, warp: Warp)
        : Array<MemoryAccess | null>
    {
        const accesses = _.range(trace.warpSize).map(() => null);
        const start = getWarpStart(trace, warp);

        for (const access of warp.accesses)
        {
            accesses[getLaneId(access, start, trace.blockDimension)] = access;
        }

        return accesses;
    }
}
