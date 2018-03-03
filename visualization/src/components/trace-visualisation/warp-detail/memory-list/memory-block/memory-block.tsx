import React from 'react';
import {PureComponent} from 'react';
import {MemoryAllocation} from '../../../../../lib/profile/memory-allocation';
import {
    addressToNum, checkIntersection, getAddressRangeSize,
    getAllocationAddressRange, numToAddress, getSelectionRange
} from '../../../../../lib/profile/address';
import {AddressRange, WarpAddressSelection} from '../../../../../lib/trace/selection';
import {zoom} from 'd3-zoom';
import GridLayout from 'd3-v4-grid';
import {select} from 'd3-selection';
import {range} from 'd3-array';
import * as d3 from 'd3';
import {formatAddressSpace, formatByteSize} from '../../../../../lib/util/format';
import {Badge, Card, CardBody} from 'reactstrap';
import CardHeader from 'reactstrap/lib/CardHeader';
import {getFilename} from '../../../../../lib/util/string';
import styled from 'styled-components';

interface Props
{
    allocation: MemoryAllocation;
    rangeSelections: WarpAddressSelection[];
    onMemorySelect: (memorySelection: AddressRange[]) => void;
}

const selectedBlock = 'rgb(220, 0, 0)';
const activeBlock = 'rgb(65, 105, 225)';

const Wrapper = styled(Card)`
  width: 100%;
`;
const Header = styled(CardHeader)`
  padding: 10px;
`;
const Body = styled(CardBody)`
  padding: 10px;
`;
const MemoryBadge = styled(Badge)`
  margin-right: 5px;
`;
const BadgeAddress = MemoryBadge.extend`
  background-color: #337AB7;
`;
const BadgeDecl = MemoryBadge.extend`
  background-color: #B353B7;
`;
const BadgeSource = MemoryBadge.extend`
  background-color: #1AB717;
`;

export class MemoryBlock extends PureComponent<Props>
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

        const effectiveRange = this.calculateRange();
        const size = getAddressRangeSize(effectiveRange);

        const dim = 16;
        const elements = dim * dim;
        const blockSize = Math.max(4, Math.ceil(size / elements));

        // label
        select(this.blockWrapper)
            .select('.block-label')
            .text(`${effectiveRange.from} (block size ${blockSize})`);

        const grid = GridLayout()
            .data(range(elements).map((index: number) => ({ index })))
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
            .attr('stroke-width', '0.5')
            .attr('fill', activeBlock)
            .on('mouseenter', (data: {index: number}) => {
                const blockFrom = start.add(data.index * blockSize);
                const blockTo = blockFrom.add(blockSize);
                this.props.onMemorySelect([{
                    from: numToAddress(blockFrom),
                    to: numToAddress(blockTo)
                }]);
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
                    return selectedBlock;
                }
                else return activeBlock;
            });
        }
        else blocks.attr('fill', activeBlock);
    }

    render()
    {
        return (
            <Wrapper>
                <Header>
                    {this.createLabel(this.props.allocation)}
                </Header>
                <Body>
                    <div ref={(wrapper) => this.blockWrapper = wrapper}>
                        <div className='block-label' />
                        <svg width={'100%'}>
                            <g className='blocks' />
                        </svg>
                    </div>
                </Body>
            </Wrapper>
        );
    }

    createLabel = (allocation: MemoryAllocation): JSX.Element =>
    {
        return (
            <>
                <BadgeDecl>{this.createVarDecl(allocation)}</BadgeDecl>
                <BadgeAddress>{allocation.address}</BadgeAddress>
                <MemoryBadge>{formatAddressSpace(allocation.space)}</MemoryBadge>
                <BadgeSource>{getFilename(allocation.location)}</BadgeSource>
            </>
        );
    }
    createVarDecl = (allocation: MemoryAllocation): JSX.Element =>
    {
        const {size, elementSize, type, name} = allocation;
        if (size && elementSize && type && name)
        {
            return <>{type} {name}[{size / elementSize}]</>;
        }
        return <>{formatByteSize(size)} of {type}</>;
    }

    calculateRange = (): AddressRange =>
    {
        return getSelectionRange(getAllocationAddressRange(this.props.allocation), this.props.rangeSelections);
    }
}
