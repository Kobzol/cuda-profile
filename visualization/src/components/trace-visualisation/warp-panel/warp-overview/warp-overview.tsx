import React, {PureComponent} from 'react';
import {Warp} from '../../../../lib/profile/warp';
import {Button} from 'react-bootstrap';
import GridLayout from 'd3-v4-grid';
import {range} from 'd3-array';
import {select} from 'd3-selection';
import {formatDim3} from '../../../../lib/util/format';
import * as _ from 'lodash';
import * as d3 from 'd3';

import './warp-overview.scss';

interface Props
{
    warps: Warp[];
    selectedWarps: Warp[];
    onWarpSelect: (warp: Warp[]) => void;
}

interface State
{
    limit: number;
}

export class WarpOverview extends PureComponent<Props, State>
{
    private blockWrapper: HTMLDivElement;

    constructor(props: Props)
    {
        super(props);

        this.state = {
            limit: 100
        };
    }

    componentDidMount()
    {
        this.renderd3();
    }

    componentDidUpdate()
    {
        this.renderd3();
    }

    renderd3()
    {
        const svg = select(this.blockWrapper).select('svg');
        const size = Math.min(this.props.warps.length, this.state.limit);
        const grid = GridLayout()
            .data(range(size).map((index: number) => ({ index })))
            .nodeSize([20, 20])
            .cols(10)
            .padding([4, 4])
            .bands(true);
        grid.layout();

        const nodeSize = grid.nodeSize();
        const group = svg.select('.blocks');

        let blocks = group
            .selectAll('rect')
            .data(grid.nodes());

        const props = (selection: typeof blocks) => {
            selection
                .attr('x', (d: {x: number}) => d.x)
                .attr('y', (d: {y: number}) => d.y)
                .attr('width', nodeSize[0])
                .attr('height', nodeSize[1])
                .attr('fill', ({index}: {index: number}) => {
                    const warp = this.props.warps[index];
                    if (_.includes(this.props.selectedWarps, warp))
                    {
                        return 'rgb(0, 0, 255)';
                    }
                    else return 'rgb(255, 255, 255)';
                });
        };
        const textProps = (selection: typeof blocks) => {
            selection.text(({index}: {index: number}) => {
                const warp = this.props.warps[index];

                return `${formatDim3(warp.blockIdx)}`;
            });
        };

        blocks.call(props);
        blocks.select('title').call(textProps);
        blocks
            .enter()
            .append('rect')
            .call(props)
            .attr('stroke', 'rgb(0, 0, 0)')
            .attr('stroke-width', '1')
            .on('click', ({index}: {index: number}) => {
                const warp = this.props.warps[index];
                if (d3.event.ctrlKey)
                {
                    this.handleSelectAdd(warp);
                }
                else this.handleSelect(warp);
            })
            .append('title')
            .call(textProps);

        blocks
            .exit()
            .remove();
    }

    render()
    {
        const increaseLimit = this.state.limit < this.props.warps.length;

        return (
            <div ref={ref => this.blockWrapper = ref} className='warp-overview'>
                <svg width={'100%'}>
                    <g className='blocks' />
                </svg>
                {increaseLimit && <Button onClick={this.increaseLimit}>Load more</Button>}
            </div>
        );
    }

    increaseLimit = () =>
    {
        this.setState((state: State) => ({
            ...state,
            limit: state.limit + 100
        }));
    }

    handleSelect = (warp: Warp) =>
    {
        if (_.includes(this.props.selectedWarps, warp))
        {
            this.props.onWarpSelect(this.props.selectedWarps.filter(w => w !== warp));
        }
        else this.props.onWarpSelect([warp]);
    }
    handleSelectAdd = (warp: Warp) =>
    {
        if (_.includes(this.props.selectedWarps, warp))
        {
            this.props.onWarpSelect(this.props.selectedWarps.filter(w => w !== warp));
        }
        else this.props.onWarpSelect([...this.props.selectedWarps, warp]);
    }
}
