import React, {PureComponent} from 'react';
import {Trace} from '../../../lib/profile/trace';
import {Warp} from '../../../lib/profile/warp';
import {AddressRange, WarpAddressSelection} from '../../../lib/trace/selection';
import {Panel, Tab, Tabs} from 'react-bootstrap';
import {MemoryConflictTable} from './memory-conflict-table/memory-conflict-table';
import {BankConflictTable} from './bank-conflict-table/bank-conflict-table';

import style from './warp-detail.scss';
import {MemoryList} from '../memory-list/memory-list';

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

export class WarpDetail extends PureComponent<Props, State>
{
    constructor(props: Props)
    {
        super(props);

        this.state = {
            activeTab: 0
        };
    }

    render()
    {
        return (
            <Panel header='Detail'  bsStyle='primary'>
                <Tabs activeKey={this.state.activeTab}
                      animation={false}
                      onSelect={this.handleSelect}
                      id='warp-detail'>
                    <Tab eventKey={0} title='Memory conflicts'>
                        <MemoryConflictTable
                            trace={this.props.trace}
                            warps={this.props.warps}
                            onMemorySelect={this.props.onMemorySelect} />
                    </Tab>
                    <Tab eventKey={1} title='Bank conflicts'>
                        <BankConflictTable
                            trace={this.props.trace}
                            warps={this.props.warps}
                            onMemorySelect={this.props.onMemorySelect} />
                    </Tab>
                    <Tab eventKey={2} title='Memory map'>
                        <MemoryList
                            allocations={this.props.trace.allocations}
                            rangeSelections={this.props.rangeSelections}
                            onMemorySelect={this.props.onMemorySelect} />
                    </Tab>
                </Tabs>
            </Panel>
        );
    }

    handleSelect = (e: React.MouseEvent<{}>) =>
    {
        this.setState(() => ({
            activeTab: e as {} as number
        }));
    }
}
