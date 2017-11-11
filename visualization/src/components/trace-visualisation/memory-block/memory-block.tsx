import * as React from 'react';
import {PureComponent} from 'react';
import {MemoryAllocation} from '../../../lib/profile/memory-allocation';
import {
    addressToNum,
    checkIntersection, clampAddressRange, getAddressRangeSize,
    getAllocationAddressRange, numToAddress
} from '../../../lib/profile/address';
import {AddressRange, WarpAddressSelection} from '../../../lib/trace/selection';
import {zoom} from 'd3-zoom';
import GridLayout from 'd3-v4-grid';
import {select} from 'd3-selection';
import {range} from 'd3-array';
import * as d3 from 'd3';
import {formatAddressSpace, formatByteSize} from '../../../lib/util/format';
import {Panel} from 'react-bootstrap';

import './memory-block.css';


interface Props
{
    allocation: MemoryAllocation;
    rangeSelections: WarpAddressSelection[];
    onMemorySelect: (memorySelection: AddressRange) => void;
}

interface State
{

}

export class MemoryBlock extends PureComponent<Props, State>
{
    private blockWrapper: HTMLDivElement = null;

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
        const width = (svg.node() as Element).getBoundingClientRect().width;
        const height = (svg.node() as Element).getBoundingClientRect().height;

        const allocRange = getAllocationAddressRange(this.props.allocation);
        const effectiveRange = clampAddressRange(this.props.rangeSelections.length > 0 ?
            this.props.rangeSelections[0].warpRange : allocRange, allocRange);
        const size = getAddressRangeSize(effectiveRange);

        const dim = 16;
        const elements = dim * dim;
        const blockSize = Math.max(4, Math.ceil(size / elements));

        // label
        select(this.blockWrapper)
            .select('.block-label')
            .text(`${effectiveRange.from} (block size ${blockSize})`);

        const grid = GridLayout()
            .data(range(elements).map(index => ({ index })))
            .bands(true)
            .size([width, height]);
        grid.layout();

        const nodeSize = grid.nodeSize();
        const group = svg.select('.blocks');

        const zoomer = zoom()
            .scaleExtent([1, 4])
            .translateExtent([
                [0, 0], // [-nodeSize[0], -nodeSize[1]],
                [width, height] // [width + nodeSize[0], height + nodeSize[1]]
            ]);
        zoomer.on('zoom', () => {
            group.attr('transform', d3.event.transform);
        });

        // zoom
        group.style('pointer-events', 'all')
            .call(zoomer);

        const start = addressToNum(effectiveRange.from);

        let blocks = group
            .selectAll('rect')
            .data(grid.nodes());

        const props = (selection: typeof blocks) => {
            selection
                .attr('x', (d: {x: number}) => d.x)
                .attr('y', (d: {y: number}) => d.y)
                .attr('width', nodeSize[0])
                .attr('height', nodeSize[1]);
        };
        const textProps = (selection: typeof blocks) => {
            selection.text((data: {index: number}) => {
                const blockFrom = start.add(data.index * blockSize);
                const blockTo = blockFrom.add(blockSize);
                return `${numToAddress(blockFrom)} - ${numToAddress(blockTo)} (${blockSize} bytes)`;
            });
        };

        blocks.call(props);
        blocks.select('title').call(textProps);
        blocks
            .enter()
            .append('rect')
            .call(props)
            .attr('stroke', 'rgb(0, 0, 0)')
            .attr('stroke-width', '0.2')
            .attr('fill', 'rgb(0, 0, 255)')
            .on('mouseenter', (data: {index: number}) => {
                const blockFrom = start.add(data.index * blockSize);
                const blockTo = blockFrom.add(blockSize);
                this.props.onMemorySelect({
                    from: numToAddress(blockFrom),
                    to: numToAddress(blockTo)
                });
            })
            .on('mouseleave', () => {
                this.props.onMemorySelect(null);
            })
            .append('title')
            .call(textProps);

        blocks
            .exit()
            .remove();

        if (this.props.rangeSelections.length > 0)
        {
            const rangeFrom = addressToNum(this.props.rangeSelections[0].threadRange.from);
            const rangeTo = addressToNum(this.props.rangeSelections[0].threadRange.to);

            blocks.attr('fill', (data: {index: number}) => {
                const blockFrom = start.add(data.index * blockSize);
                const blockTo = blockFrom.add(blockSize);

                if (checkIntersection(rangeFrom, rangeTo, blockFrom, blockTo))
                {
                    return 'rgb(255, 0, 0)';
                }
                else return 'rgb(0, 0, 255)';
            });
        }
        else blocks.attr('fill', 'rgb(0, 0, 255)');
    }

    render()
    {
        return (
            <Panel className='memory-block' id='memory-block'
                   header={this.createLabel(this.props.allocation)}>
                <div className='block-wrapper' ref={(wrapper) => this.blockWrapper = wrapper}>
                    <div className='block-label' />
                    <svg width={'100%'}>
                        <g className='blocks' />
                    </svg>
                </div>
            </Panel>
        );
    }

    createLabel = (allocation: MemoryAllocation): string =>
    {
        const {size, address, space, type, name, location} = allocation;
        let label = `${formatByteSize(size)} of ${type} allocated at ${address} (${formatAddressSpace(space)} space)`;
        if (name !== '')
        {
            label += `, variable ${name}`;
        }
        if (location !== '')
        {
            label += `, at ${location}`;
        }

        return label;
    }
}
