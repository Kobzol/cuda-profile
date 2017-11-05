import React, {PureComponent} from 'react';
import {MemoryAccess} from '../../../../lib/profile/memory-access';
import {Trace} from '../../../../lib/profile/trace';
import {createBlockSelector} from './grid-data';
import {AddressRange} from '../../../../lib/trace/selection';
import {getBlockId, Warp} from '../../../../lib/profile/warp';
import {Thread} from './thread';
import {WarpAddressSelection} from '../../../../lib/trace/selection';
import {getAccessAddressRange} from '../../../../lib/profile/address';
import {Selector} from 'reselect';
import {Dictionary} from 'lodash';
import GridLayout from 'd3-v4-grid';
import * as _ from 'lodash';

import './warp-grid.css';

interface Props
{
    trace: Trace;
    warp: Warp;
    memorySelection: AddressRange;
    canvasDimensions: { width: number, height: number };
    selectRange: (range: WarpAddressSelection) => void;
}
interface State
{
    blockMapSelector: Selector<Warp, Dictionary<MemoryAccess>>;
}

const dims = {
    rows: 4,
    cols: 8
};

export class WarpGrid extends PureComponent<Props, State>
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
            { rows: dims.rows, cols: dims.cols },
            { width, height }
        );
        const nodeSize = {
            width: layout.nodeSize()[0],
            height: layout.nodeSize()[1]
        };

        const grid = this.renderGrid(layout.nodes(), nodeSize);

        return (
            <div className='warp-grid'>
                {this.renderLabel(this.props.warp, this.props.trace)}
                <svg width={width}
                     height={height}
                     viewBox={`0 0 ${width} ${height}`}>
                    <g>{grid}</g>
                </svg>
            </div>
        );
    }

    renderLabel = (warp: Warp, trace: Trace): JSX.Element =>
    {
        const blockId = getBlockId(warp.blockIdx, trace.gridDimension);

        return (
            <div>{`Warp ${warp.id} in block ${blockId} (slot ${warp.slot})`}</div>
        );
    }

    renderGrid = (nodes: Array<{x: number, y: number}>,
                  nodeSize: {width: number, height: number}): JSX.Element[] =>
    {
        const grid: JSX.Element[] = [];
        const width = dims.cols;
        const height = dims.rows;

        const warp = this.props.warp;
        const accesses = this.createWarpAccesses(this.props.trace, warp);
        for (let y = 0; y < height; y++)
        {
            for (let x = 0; x < width; x++)
            {
                const index = y * (width) + x;
                const access = accesses[index];

                grid.push(
                    <Thread
                        key={index}
                        x={nodes[index].x}
                        y={nodes[index].y}
                        width={nodeSize.width}
                        height={nodeSize.height}
                        warp={warp}
                        access={access}
                        memorySelection={this.props.memorySelection}
                        onSelectChanged={this.handleRangeSelectChange} />
                );
            }
        }

        return grid;
    }

    handleRangeSelectChange = (range: AddressRange) =>
    {
        if (range !== null)
        {
            this.props.selectRange({
                warpRange: getAccessAddressRange(this.props.warp.accesses, this.props.warp.size),
                threadRange: range
            });
        }
        else this.props.selectRange(null);
    }

    calculateLayout = (gridSize: {rows: number, cols: number},
                       canvasSize: {width: number, height: number}) =>
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

    createWarpAccesses = (trace: Trace, warp: Warp)
        : Array<MemoryAccess | null> =>
    {
        const accesses = _.range(trace.warpSize).map(() => null);
        for (const access of warp.accesses)
        {
            accesses[access.id] = access;
        }
        return accesses;
    }
}
