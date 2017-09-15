import React, {ChangeEvent, DragEvent, PureComponent} from "react";
import {connect} from "react-redux";
import {TraceFile} from "../../lib/trace/trace-file";
import {AppState} from "../../state/reducers";
import {loadFile} from "../../lib/trace/actions";

interface StateProps
{
    files: TraceFile[];
}

interface DispatchProps
{
    loadFile: (file: File) => any;
}

class TraceLoaderComponent extends PureComponent<StateProps & DispatchProps>
{
    render()
    {
        return (
            <div>
                <input type="file" multiple={true} onChange={this.handleTraceChange} onDrop={this.handleTraceDrop} />
                {this.props.files.map(this.renderFile)}
            </div>
        );
    }

    renderFile = (file: TraceFile): JSX.Element =>
    {
        return (
            <div key={file.id}>
                <span>{file.name}, loading: {file.loading ? "true" : "false"}</span>
            </div>
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
    files: state.trace.files
}), ({
    loadFile: loadFile.started
}))(TraceLoaderComponent);
