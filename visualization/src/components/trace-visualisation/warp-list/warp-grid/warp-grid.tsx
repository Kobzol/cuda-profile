import React, {PureComponent} from 'react';
import {MemoryAccess} from '../../../../lib/profile/memory-access';
import {Trace} from '../../../../lib/profile/trace';
import {createBlockSelector} from './grid-data';
import {AddressRange, WarpAccess} from '../../../../lib/trace/selection';
import {getWarpId, Warp, AccessType} from '../../../../lib/profile/warp';
import {Thread} from './thread';
import {Selector} from 'reselect';
import {Badge as BsBadge, Button, ButtonGroup} from 'reactstrap';
import {formatAccessType, formatAddressSpace, formatDim3} from '../../../../lib/util/format';
import MdHourglassEmpty from 'react-icons/lib/md/hourglass-empty';
import MdClose from 'react-icons/lib/md/close';
import styled from 'styled-components';
import {BlockParams, SVGGrid} from '../../svg-grid/svg-grid';
import {Dictionary, range, findIndex, equals} from 'ramda';

interface Props
{
    trace: Trace;
    warp: Warp;
    index: number;
    memorySelection: AddressRange[];
    canvasDimensions: { width: number, height: number };
    selectedAccesses: WarpAccess[];
    onAccessSelectionChange(access: WarpAccess, active: boolean): void;
    onDeselect(warp: Warp): void;
    onSelectAllWarpAccesses(warp: Warp): void;
}
interface State
{
    blockMapSelector: Selector<Warp, Dictionary<MemoryAccess>>;
}

const WarpWrapper = styled.div`
  padding: 4px;
  background-color: #DDDDDD;
  border: 1px solid #888888;
  border-radius: 5px;
  margin: 0 5px 5px 0;
`;
const WarpTitle = styled.div`
  display: flex;
  margin-bottom: 3px;
`;
const Content = styled.div`
  display: flex;
`;
const Buttons = styled(ButtonGroup)`
  flex-grow: 1;
  justify-content: flex-end;
`;
const WarpButton = styled(Button)`
  padding: 0;
  line-height: 1;
`;
const Badge = styled(BsBadge)`
  margin-right: 5px;
  font-size: 12px;
`;
const BadgeBlock = Badge.extend`
  background-color: #337AB7;
`;
const BadgeSize = Badge.extend`
  background-color: #800080;
`;
const BadgeRead = Badge.extend`
  background-color: #006400;
`;
const BadgeWrite = Badge.extend`
  background-color: #8B0000;
`;

export class WarpGrid extends PureComponent<Props, State>
{
    state: State = {
        blockMapSelector: createBlockSelector()
    };

    render()
    {
        const {width, height} = this.props.canvasDimensions;
        return (
            <WarpWrapper>
                {this.renderLabel(this.props.warp, this.props.trace)}
                <Content>
                    <SVGGrid width={width}
                             height={height}
                             rows={4}
                             cols={8}
                             renderItem={this.renderThread}
                             {...{
                                 memorySelection: this.props.memorySelection,
                                 selectedAccesses: this.props.selectedAccesses
                             }} />
                </Content>
            </WarpWrapper>
        );
    }
    renderThread = (params: BlockParams): JSX.Element =>
    {
        const warp = this.props.warp;
        const accesses = this.createWarpAccesses(this.props.trace, warp);
        const access = accesses[params.index];

        const selectedIndex = findIndex(warpAccess =>
            warpAccess.warp === warp &&
            equals(warpAccess.access, access), this.props.selectedAccesses
        );
        const selected = selectedIndex !== -1;

        return (
            <Thread
                x={params.x}
                y={params.y}
                width={params.width}
                height={params.height}
                warp={warp}
                access={access}
                selected={selected}
                selectedIndex={selectedIndex}
                memorySelection={this.props.memorySelection}
                onSelectChanged={this.props.onAccessSelectionChange} />
        );
    }

    renderLabel = (warp: Warp, trace: Trace): JSX.Element =>
    {
        const title = `Warp id ${getWarpId(warp.accesses[0].threadIdx, trace.warpSize, trace.blockDimension)}, ` +
        `block ${formatDim3(warp.blockIdx)}, ${warp.size} bytes ${formatAccessType(warp.accessType)}, at ` +
        `${warp.timestamp}`;

        const AccessBadge = warp.accessType === AccessType.Read ? BadgeRead : BadgeWrite;
        return (
            <WarpTitle title={title}>
                <BadgeBlock>
                    {formatDim3(warp.blockIdx)}
                </BadgeBlock>
                <AccessBadge>
                    {formatAccessType(warp.accessType)}
                </AccessBadge>
                <BadgeSize>
                    {warp.size} b
                </BadgeSize>
                <Badge>
                    {formatAddressSpace(warp.space)}
                </Badge>
                <Buttons>
                    <WarpButton onClick={this.selectAllWarpAccesses}
                                size='small'
                                title='Select all accesses of this warp'>
                        <MdHourglassEmpty />
                    </WarpButton>
                    <WarpButton onClick={this.deselect}
                                size='small'
                                title='Deselect'>
                        <MdClose />
                    </WarpButton>
                </Buttons>
            </WarpTitle>
        );
    }

    createWarpAccesses = (trace: Trace, warp: Warp)
        : Array<MemoryAccess | null> =>
    {
        const accesses = range(0, trace.warpSize).map(() => null);
        for (const access of warp.accesses)
        {
            accesses[access.id] = access;
        }
        return accesses;
    }

    selectAllWarpAccesses = () =>
    {
        this.props.onSelectAllWarpAccesses(this.props.warp);
    }

    deselect = () =>
    {
        this.props.onDeselect(this.props.warp);
    }
}
