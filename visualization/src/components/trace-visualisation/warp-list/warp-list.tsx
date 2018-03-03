import React, {PureComponent} from 'react';
import {Trace} from '../../../lib/profile/trace';
import {Warp} from '../../../lib/profile/warp';
import {WarpGrid} from './warp-grid/warp-grid';
import {AddressRange, WarpAddressSelection} from '../../../lib/trace/selection';
import {getAccessesAddressRange} from '../../../lib/profile/address';
import {Button, Card, CardHeader, CardBody} from 'reactstrap';
import styled from 'styled-components';
import {chain} from 'ramda';

interface Props
{
    trace: Trace;
    warps: Warp[];
    memorySelection: AddressRange[];
    selectRange(range: WarpAddressSelection): void;
    deselect(warp: Warp): void;
    clearSelection(): void;
    selectAllWarpAccesses(warp: Warp): void;
}

const Wrapper = styled(Card)`
  min-width: 260px;
`;
const Header = styled(CardHeader)`
  display: flex;
  justify-content: space-between;
  align-items: center;
`;
const WarpBody = styled(CardBody)`
  display: flex;
  flex-wrap: wrap;
  padding: 10px;
`;

export class WarpList extends PureComponent<Props>
{
    render()
    {
        return (
            <Wrapper>
                <Header>
                    Selected accesses
                    {this.props.warps.length > 0 &&
                        <Button title='Clear selection'
                                size='small'
                                onClick={this.props.clearSelection}>
                            Clear selection
                        </Button>
                    }
                </Header>
                <WarpBody>
                    {this.props.warps.length === 0 && 'No warps selected'}
                    {this.props.warps.map((warp, index) =>
                        <WarpGrid
                            key={warp.key}
                            index={index}
                            trace={this.props.trace}
                            warp={warp}
                            canvasDimensions={{width: 220, height: 50}}
                            selectRange={this.handleRangeSelect}
                            memorySelection={this.props.memorySelection}
                            selectionEnabled={true}
                            deselect={this.props.deselect}
                            selectAllWarpAccesses={this.props.selectAllWarpAccesses} />
                    )}
                </WarpBody>
            </Wrapper>
        );
    }

    handleRangeSelect = (range: WarpAddressSelection) =>
    {
        if (range !== null)
        {
            range.warpRange = getAccessesAddressRange(chain(warp => warp.accesses, this.props.warps),
                this.props.warps[0].size);
        }
        this.props.selectRange(range);
    }
}
