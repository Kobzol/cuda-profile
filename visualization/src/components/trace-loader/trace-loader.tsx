import React, {ChangeEvent, DragEvent, PureComponent} from "react";
import {connect} from "react-redux";
import {TraceFile} from "../../lib/trace/trace-file";
import {AppState} from "../../state/reducers";
import {loadTraceFile} from "../../lib/trace/actions";
import {validTraceFiles} from "../../lib/trace/reducer";
import {push} from "react-router-redux";

interface StateProps
{
    files: TraceFile[];
    validTraceFiles: TraceFile[];
}

interface DispatchProps
{
    loadFile: (file: File) => any;
    navigateToKernelView: () => any;
}

class TraceLoaderComponent extends PureComponent<StateProps & DispatchProps>
{
    render()
    {
        return (
            <div>
                <input type="file" multiple={true} onChange={this.handleTraceChange} onDrop={this.handleTraceDrop} />
                <ul>
                    {this.props.files.map(this.renderFile)}
                </ul>
                <button
                    disabled={this.props.validTraceFiles.length < 1}
                    onClick={this.props.navigateToKernelView}
                >Load trace</button>
            </div>
        );
    }

    renderFile = (file: TraceFile): JSX.Element =>
    {
        return (
            <li key={file.id}>
                <span>{file.name}, loading: {file.loading ? "true" : "false"}, error: {file.error}</span>
            </li>
        );
    };

    handleTraceChange = (event: ChangeEvent<HTMLInputElement>) =>
    {
        for (let i = 0; i < event.target.files.length; i++)
        {
            this.props.loadFile(event.target.files[i]);
        }
    };
    handleTraceDrop = (event: DragEvent<HTMLInputElement>) =>
    {
        const files = event.dataTransfer.files;
        // add files
    };
}

export const TraceLoader = connect<StateProps, DispatchProps, {}>((state: AppState) => ({
    files: state.trace.files,
    validTraceFiles: validTraceFiles(state)
}), ({
    loadFile: loadTraceFile.started,
    navigateToKernelView: () => push('/kernel-launches')
}))(TraceLoaderComponent);
