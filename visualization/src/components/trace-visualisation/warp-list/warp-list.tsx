import React, {PureComponent} from 'react';
import {Trace} from '../../../lib/profile/trace';
import {Warp} from '../../../lib/profile/warp';
import {WarpGrid} from './warp-grid/warp-grid';
import {AddressRange, WarpAccess} from '../../../lib/trace/selection';
import {Button, Card, CardHeader, CardBody} from 'reactstrap';
import styled from 'styled-components';
import MdClear from 'react-icons/lib/md/clear';

interface Props
{
    trace: Trace;
    warps: Warp[];
    memorySelection: AddressRange[];
    selectedAccesses: WarpAccess[];
    onAccessSelectionChange(access: WarpAccess, active: boolean): void;
    onDeselect(warp: Warp): void;
    onClearSelection(): void;
    onSelectAllWarpAccesses(warp: Warp): void;
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
                                onClick={this.props.onClearSelection}>
                            <MdClear />
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
                            selectedAccesses={this.props.selectedAccesses}
                            memorySelection={this.props.memorySelection}
                            onAccessSelectionChange={this.props.onAccessSelectionChange}
                            onDeselect={this.props.onDeselect}
                            onSelectAllWarpAccesses={this.props.onSelectAllWarpAccesses} />
                    )}
                </WarpBody>
            </Wrapper>
        );
    }
}
