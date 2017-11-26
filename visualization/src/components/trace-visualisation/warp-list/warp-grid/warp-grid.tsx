import React, {PureComponent} from 'react';
import {MemoryAccess} from '../../../../lib/profile/memory-access';
import {Trace} from '../../../../lib/profile/trace';
import {createBlockSelector} from './grid-data';
import {AddressRange} from '../../../../lib/trace/selection';
import {getBlockId, getWarpId, Warp} from '../../../../lib/profile/warp';
import {Thread} from './thread';
import {WarpAddressSelection} from '../../../../lib/trace/selection';
import {getAccessesAddressRange} from '../../../../lib/profile/address';
import {Selector} from 'reselect';
import {Dictionary} from 'lodash';
import GridLayout from 'd3-v4-grid';
import * as _ from 'lodash';

import './warp-grid.scss';
import {Button, Glyphicon, Panel} from 'react-bootstrap';
import {formatAccessType, formatDim3} from '../../../../lib/util/format';

interface Props
{
    trace: Trace;
    warp: Warp;
    memorySelection: AddressRange;
    canvasDimensions: { width: number, height: number };
    selectRange: (range: WarpAddressSelection) => void;
    selectionEnabled: boolean;
    deselect: (warp: Warp) => void;
    selectAllWarpAccesses: (warp: Warp) => void;
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
            <Panel className='warp-grid' header={this.renderLabel(this.props.warp, this.props.trace)}>
                <div className='content'>
                    <svg width={width}
                         height={height}
                         viewBox={`0 0 ${width} ${height}`}>
                        <g>{grid}</g>
                    </svg>
                    <Button onClick={this.selectAllWarpAccesses}
                            title='Select all accesses of this warp'
                            className='action'>
                        <Glyphicon glyph='hourglass' />
                    </Button>
                    <Button onClick={this.deselect} title='Deselect' className='action'>
                        <Glyphicon glyph='remove' />
                    </Button>
                </div>
            </Panel>
        );
    }

    renderLabel = (warp: Warp, trace: Trace): string =>
    {
        return `Warp id ${getWarpId(warp.accesses[0].threadIdx, trace.warpSize, trace.blockDimension)},
                block ${formatDim3(warp.blockIdx)}, ${warp.size} bytes ${formatAccessType(warp.accessType)}, at
                ${warp.timestamp}`;
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
                        onSelectChanged={this.handleRangeSelectChange}
                        selectionEnabled={this.props.selectionEnabled} />
                );
            }
        }

        return grid;
    }

    handleRangeSelectChange = (range: AddressRange) =>
    {
        if (this.props.selectionEnabled)
        {
            if (range !== null)
            {
                this.props.selectRange({
                    warpRange: getAccessesAddressRange(this.props.warp.accesses, this.props.warp.size),
                    threadRange: range
                });
            }
            else this.props.selectRange(null);
        }
    }

    calculateLayout = (gridSize: {rows: number, cols: number},
                       canvasSize: {width: number, height: number}) =>
    {
        const layout = GridLayout()
            .data(_.range(gridSize.rows * gridSize.cols))
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

    selectAllWarpAccesses = () =>
    {
        this.props.selectAllWarpAccesses(this.props.warp);
    }

    deselect = () =>
    {
        this.props.deselect(this.props.warp);
    }
}
