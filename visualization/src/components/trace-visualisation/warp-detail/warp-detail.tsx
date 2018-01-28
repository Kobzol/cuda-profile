import React, {PureComponent} from 'react';
import {Trace} from '../../../lib/profile/trace';
import {Warp} from '../../../lib/profile/warp';
import {AddressRange, WarpAddressSelection} from '../../../lib/trace/selection';
import {Tab, Tabs} from 'react-bootstrap';
import {WarpConflictTable} from './warp-conflict-table/warp-conflict-table';
import {BankConflictTable} from './bank-conflict-table/bank-conflict-table';

interface Props
{
    trace: Trace;
    warps: Warp[];
    rangeSelections: WarpAddressSelection[];
    onMemorySelect: (selection: AddressRange) => void;
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
            <div>
                <h3>Detail</h3>
                <Tabs activeKey={this.state.activeTab}
                      animation={false}
                      onSelect={this.handleSelect}
                      id='warp-detail'>
                    <Tab eventKey={0} title='Memory conflicts'>
                        <WarpConflictTable
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
                    <Tab eventKey={2} title='Contiguity'>Contiguity</Tab>
                </Tabs>
            </div>
        );
    }

    handleSelect = (e: React.MouseEvent<{}>) =>
    {
        this.setState(() => ({
            activeTab: e as {} as number
        }));
    }
}
