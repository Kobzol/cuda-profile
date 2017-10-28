import React, {PureComponent} from 'react';
import {AccessType, MemoryAccess, Warp} from '../../../lib/profile/memory-access';
import {Trace} from '../../../lib/profile/trace';
import {createBlockSelector} from './grid-data';
import {Selector} from 'reselect';
import {Dictionary} from 'lodash';
import GridLayout from 'd3-v4-grid';
import * as _ from 'lodash';

import './thread-grid.css';

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
        const {width, height} = this.props.canvasDimensions;
        const layout = this.calculateLayout(
            { rows: 4, cols: 8 },
            { width, height }
        );
        const nodeSize = {
            width: layout.nodeSize()[0],
            height: layout.nodeSize()[1]
        };

        const grid = this.renderGrid(layout.nodes(), nodeSize);

        return (
            <svg className='thread-grid' width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
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

        const warp = this.props.warp;
        const {x: bx, y: by, z: bz } = warp.blockIdx;

        const accesses = this.createWarpAccesses(this.props.trace, warp);
        for (let y = 0; y < height; y++)
        {
            for (let x = 0; x < width; x++)
            {
                const index = y * (width) + x;
                const access = accesses[index];
                let label = 'Inactive thread';

                if (access !== null)
                {
                    const {x: tx, y: ty, z: tz} = access.threadIdx;
                    label = `${bz}.${by}.${bx}.${tz}.${ty}.${tx}: ${warp.size} at ${access.address}`;
                }

                grid.push(
                    <g>
                        <rect
                            key={index}
                            x={nodes[index].x}
                            y={nodes[index].y}
                            width={nodeSize.width}
                            height={nodeSize.height}
                            fill={this.getAccessColor(warp, access)}
                            stroke='rgb(0, 0, 0)'
                            strokeWidth={0.5} />
                        <title>{label}</title>
                    </g>
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
        for (const access of warp.accesses)
        {
            accesses[access.id] = access;
        }
        return accesses;
    }

    private getAccessColor(warp: Warp, access: MemoryAccess): string
    {
        if (access === null) return 'rgb(255, 255, 255)';
        if (warp.kind === AccessType.Read)
        {
            return 'rgb(255, 0, 0)';
        }
        else return 'rgb(0, 0, 255)';
    }
}
