import React, {PureComponent} from 'react';
import {connect} from 'react-redux';
import {TraceFile} from '../../lib/file-load/file';
import {buildKernels} from '../../lib/trace/actions';
import {Kernel} from '../../lib/trace/kernel';
import {AppState} from '../../state/reducers';
import {KernelComponent} from '../kernel/kernel';

interface StateProps
{
    files: TraceFile[];
    kernels: Kernel[];
}
interface DispatchProps
{
    buildKernels: (files: TraceFile[]) => {};
}

class KernelLaunchSelectorComponent extends PureComponent<StateProps & DispatchProps>
{
    componentWillMount()
    {
        this.props.buildKernels(this.props.files);
    }

    render()
    {
        return (
            <div>{this.props.kernels.map(this.renderKernel)}</div>
        );
    }

    renderKernel = (kernel: Kernel): JSX.Element =>
    {
        return (
            <KernelComponent
                key={kernel.metadata.kernel}
                kernel={kernel} />
        );
    }
}

export const KernelLaunchSelector = connect<StateProps, DispatchProps, {}>((state: AppState) => ({
    files: state.fileLoader.files,
    kernels: state.trace.kernels
}), {
    buildKernels: buildKernels.started
})(KernelLaunchSelectorComponent);
