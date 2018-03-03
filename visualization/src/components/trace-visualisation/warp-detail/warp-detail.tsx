import React, {PureComponent} from 'react';
import {Trace} from '../../../lib/profile/trace';
import {Warp} from '../../../lib/profile/warp';
import {AddressRange, WarpAddressSelection} from '../../../lib/trace/selection';
import {Card, CardHeader, CardBody} from 'reactstrap';
import {MemoryConflictTable} from './memory-conflict-table';
import {BankConflictTable} from './bank-conflict-table';
import {MemoryList} from './memory-list/memory-list';
import {Nav, NavItem, NavLink, TabContent, TabPane} from 'reactstrap';
import styled from 'styled-components';

interface Props
{
    trace: Trace;
    warps: Warp[];
    rangeSelections: WarpAddressSelection[];
    onMemorySelect: (selection: AddressRange[]) => void;
}

interface State
{
    activeTab: number;
}

const Wrapper = styled(Card)`
  margin-top: 10px;
`;
const Body = styled(CardBody)`
  padding: 10px;
`;
const TabLink = styled(NavLink)`
  cursor: pointer;
`;

export class WarpDetail extends PureComponent<Props, State>
{
    state: State = {
        activeTab: 0
    };

    render()
    {
        return (
            <Wrapper>
                <CardHeader>Detail</CardHeader>
                <Body>
                    <Nav tabs>
                        {this.renderNav(0, 'Memory conflicts')}
                        {this.renderNav(1, 'Bank conflicts')}
                        {this.renderNav(2, 'Memory map')}
                    </Nav>
                    <TabContent activeTab={this.state.activeTab}
                          id='warp-detail'>
                        <TabPane tabId={0} title='Memory conflicts'>
                            <MemoryConflictTable
                                trace={this.props.trace}
                                warps={this.props.warps}
                                onMemorySelect={this.props.onMemorySelect} />
                        </TabPane>
                        <TabPane tabId={1} title='Bank conflicts'>
                            <BankConflictTable
                                trace={this.props.trace}
                                warps={this.props.warps}
                                onMemorySelect={this.props.onMemorySelect} />
                        </TabPane>
                        <TabPane tabId={2} title='Memory map'>
                            <MemoryList
                                allocations={this.props.trace.allocations}
                                rangeSelections={this.props.rangeSelections}
                                selectedWarps={this.props.warps}
                                onMemorySelect={this.props.onMemorySelect} />
                        </TabPane>
                    </TabContent>
                </Body>
            </Wrapper>
        );
    }
    renderNav = (index: number, text: string): JSX.Element =>
    {
        return (
            <NavItem>
                <TabLink
                    active={index === this.state.activeTab}
                    onClick={() => this.selectTab(index)}
                    title={text}>
                    {text}
                </TabLink>
            </NavItem>
        );
    }

    selectTab = (activeTab: number) =>
    {
        this.setState(() => ({ activeTab }));
    }
}
