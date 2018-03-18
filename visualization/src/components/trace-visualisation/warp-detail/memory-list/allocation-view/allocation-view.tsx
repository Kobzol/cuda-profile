import React from 'react';
import {PureComponent} from 'react';
import {MemoryAllocation} from '../../../../../lib/profile/memory-allocation';
import {
    addressToNum, getAddressRangeSize, getAllocationAddressRange,
    getWarpAccessesRange, addressAddStr, createRange,
    intersects, getAccessAddressRange,
} from '../../../../../lib/profile/address';
import {AddressRange, WarpAccess} from '../../../../../lib/trace/selection';
import {formatAddressSpace, formatByteSize} from '../../../../../lib/util/format';
import {Badge, Card, CardBody} from 'reactstrap';
import CardHeader from 'reactstrap/lib/CardHeader';
import {getFilename} from '../../../../../lib/util/string';
import styled from 'styled-components';
import {BlockParams, SVGGrid} from '../../../svg-grid/svg-grid';
import {Block} from './block';
import {Dictionary, zipObj, map, addIndex} from 'ramda';
import {Warp} from '../../../../../lib/profile/warp';

interface Props
{
    allocation: MemoryAllocation;
    selectedAccesses: WarpAccess[];
    selectedWarps: Warp[];
    onMemorySelect: (memorySelection: AddressRange[]) => void;
}

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

const rows = 20;
const cols = 40;

export class AllocationView extends PureComponent<Props>
{
    /*private wrapper: HTMLDivElement = null;

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
        const svg = select(this.wrapper).select('svg');
        const width = (svg.node() as Element).getBoundingClientRect().width;
        const height = (svg.node() as Element).getBoundingClientRect().height;

        const effectiveRange = this.calculateRange();
        const size = getAddressRangeSize(effectiveRange);

        const dim = 32;
        const elements = dim * dim;
        const blockSize = Math.max(4, Math.ceil(size / elements));

        // label
        select(this.wrapper)
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
            .attr('fill', 'rgb(0, 0, 0)')
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

        if (this.props.selectedAccesses.length > 0)
        {
            blocks.attr('fill', (data: {index: number}) => {
                const blockFrom = start.add(data.index * blockSize);
                const blockTo = blockFrom.add(blockSize);

                if (this.props.selectedAccesses.find(a => checkIntersection(
                    addressToNum(a.access.address),
                    addressToNum(addressAddStr(a.access.address, a.warp.size)),
                        blockFrom,
                        blockTo)) !== undefined)
                {
                    return 'rgb(255, 0, 0)';
                }

                return 'rgb(255, 255, 255)';
            });
        }
        else blocks.attr('fill', 'rgb(255, 255, 255)');
    }*/

    render()
    {
        /*return <div ref={(wrapper) => this.wrapper = wrapper} style={{width: '100%'}}>
                        <div className='block-label' />
                        <svg width={'100%'}>
                            <g className='blocks' />
                        </svg>
                    </div>;*/

        const addressRange = this.calculateRange();
        const blockSize = this.calculateBlockSize(addressRange);
        const indexedAccesses = addIndex(map)((access, index) => ({
            access,
            index
        }), this.props.selectedAccesses);
        const intersectedAccesses = indexedAccesses.filter(a =>
            intersects(getAccessAddressRange(a.access.access), addressRange)
        );
        const indices = intersectedAccesses.map(a => addressToNum(a.access.access.address)
            .subtract(addressToNum(addressRange.from))
            .divide(blockSize).toJSNumber().toString()
        );
        const accessMap = zipObj(indices, intersectedAccesses);

        return (
            <Wrapper>
                <Header>
                    {this.renderLabel(this.props.allocation, addressRange)}
                </Header>
                <Body>
                    <SVGGrid
                        width={1200}
                        height={240}
                        rows={rows}
                        cols={cols}
                        renderItem={(params) => this.renderBlock(params, addressRange, blockSize, accessMap)} />
                </Body>
            </Wrapper>
        );
    }
    renderBlock = (params: BlockParams, allocRange: AddressRange, blockSize: number,
                   accessMap: Dictionary<{access: WarpAccess, index: number}>): JSX.Element =>
    {
        const start = addressAddStr(allocRange.from, params.index * blockSize);
        const end = addressAddStr(start, blockSize);
        const addressRange = createRange(addressToNum(start), addressToNum(end));
        const index = params.index.toString();
        const indexedAccess = accessMap.hasOwnProperty(index) ? accessMap[index] : null;
        const access = indexedAccess !== null ? indexedAccess.access : null;
        const accessIndex = indexedAccess !== null ? indexedAccess.index : -1;

        return (
            <Block
                key={`${start}-${end}`}
                x={params.x}
                y={params.y}
                width={params.width}
                height={params.height}
                range={addressRange}
                access={access}
                accessIndex={accessIndex}
                onMemorySelect={this.handleMemorySelect} />
        );
    }
    renderLabel = (allocation: MemoryAllocation, addressRange: AddressRange): JSX.Element =>
    {
        return (
            <>
                <BadgeDecl>{this.renderAllocDeclaration(allocation)}</BadgeDecl>
                <BadgeAddress>{allocation.address}</BadgeAddress>
                <MemoryBadge>{formatAddressSpace(allocation.space)}</MemoryBadge>
                <BadgeSource>{getFilename(allocation.location)}</BadgeSource>
                <MemoryBadge>{addressRange.from} - {addressRange.to}</MemoryBadge>
                <MemoryBadge>one square is {this.calculateBlockSize(addressRange)} bytes</MemoryBadge>
            </>
        );
    }
    renderAllocDeclaration = (allocation: MemoryAllocation): JSX.Element =>
    {
        const {size, elementSize, type, name} = allocation;
        if (size && elementSize && type && name)
        {
            return <>{type} {name}[{size / elementSize}]</>;
        }
        return <>{formatByteSize(size)} of {type}</>;
    }

    handleMemorySelect = (addressRange: AddressRange) =>
    {
        if (addressRange === null)
        {
            this.props.onMemorySelect([]);
        }
        else this.props.onMemorySelect([addressRange]);
    }

    calculateBlockSize = (addressRange: AddressRange): number =>
    {
        const elementCount = rows * cols;
        const size = getAddressRangeSize(addressRange);
        const blockSize = Math.ceil(size / elementCount);
        const clamped = Math.max(4, blockSize);
        return Math.ceil(clamped);
    }

    calculateRange = (): AddressRange =>
    {
        return getWarpAccessesRange(getAllocationAddressRange(this.props.allocation), this.props.selectedAccesses);
        // return getWarpsRange(getAllocationAddressRange(this.props.allocation), this.props.selectedWarps);
    }
}
