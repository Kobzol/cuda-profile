import * as React from 'react';
import {PureComponent} from 'react';
import {MemoryAllocation} from '../../../lib/profile/memory-allocation';
import {
    checkIntersection, clampAddressRange, getAddressRangeSize,
    getAllocationAddressRange
} from '../../../lib/profile/address';
import {WarpAddressSelection} from '../../../lib/trace/selection';
import {zoom} from 'd3-zoom';
import GridLayout from 'd3-v4-grid';
import {select} from 'd3-selection';
import {range} from 'd3-array';
import * as d3 from 'd3';
import * as bigInt from 'big-integer';


interface Props
{
    allocation: MemoryAllocation;
    rangeSelections: WarpAddressSelection[];
}

interface State
{

}

export class MemoryBlock extends PureComponent<Props, State>
{
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
        const container = select('#memory-block');

        const width = (container.node() as any).getBoundingClientRect().width;
        const height = (container.node() as any).getBoundingClientRect().height;

        const svg = container.select('svg')
            .attr('width', width)
            .attr('height', height);

        const allocRange = getAllocationAddressRange(this.props.allocation);
        const effectiveRange = clampAddressRange(this.props.rangeSelections.length > 0 ?
            this.props.rangeSelections[0].warpRange : allocRange, allocRange);
        const size = getAddressRangeSize(effectiveRange);

        const dim = 16;
        const elements = dim * dim;
        const blockSize = Math.ceil(size / elements);

        const grid = GridLayout()
            .data(range(elements).map(index => ({ index })))
            .bands(true)
            .size([width, height]);
        grid.layout();

        const nodeSize = grid.nodeSize();
        const g = svg.select('.block-wrapper');

        const z = zoom()
            .scaleExtent([1, 4])
            .translateExtent([
                [0, 0], // [-nodeSize[0], -nodeSize[1]],
                [width, height] // [width + nodeSize[0], height + nodeSize[1]]
            ]);
        z.on('zoom', () => {
            g.attr('transform', d3.event.transform);
        });

        const zoomWrapper = svg.select('.zoom-wrapper')
            .attr('width', width)
            .attr('height', height)
            .style('fill', 'none')
            .style('pointer-events', 'all')
            .call(z);

        let example = g
            .selectAll('rect')
            .data(grid.nodes())
            .attr('x', (d: {x: number}) => d.x)
            .attr('y', (d: {y: number}) => d.y)
            .attr('width', nodeSize[0])
            .attr('height', nodeSize[1]);

        let enter = example
            .enter()
            .append('rect')
            .attr('x', (d: {x: number}) => d.x)
            .attr('y', (d: {y: number}) => d.y)
            .attr('width', nodeSize[0])
            .attr('height', nodeSize[1])
            .attr('stroke', 'rgb(0, 0, 0)')
            .attr('stroke-width', '0.2')
            .attr('fill', 'rgb(0, 0, 255)');

        let exit = example
            .exit()
            .remove();

        if (this.props.rangeSelections.length > 0)
        {
            const start = bigInt(effectiveRange.from.substr(2), 16);
            const rangeFrom = bigInt(this.props.rangeSelections[0].threadRange.from.substr(2), 16);
            const rangeTo = bigInt(this.props.rangeSelections[0].threadRange.to.substr(2), 16);

            example.attr('fill', (data: {index: number}) => {
                const blockFrom = start.add(data.index * blockSize);
                const blockTo = blockFrom.add(blockSize);

                if (checkIntersection(rangeFrom, rangeTo, blockFrom, blockTo))
                {
                    return 'rgb(255, 0, 0)';
                }
                else return 'rgb(0, 0, 255)';
            });
        }
        else example.attr('fill', 'rgb(0, 0, 255)');
    }

    render()
    {
        return (
            <div id='memory-block' style={{
                width: '600px',
                height: '120px'
            }}>
                <svg>
                    <g className='block-wrapper' />
                    <rect className='zoom-wrapper' />
                </svg>
            </div>
        );
    }
}
