import React, {ChangeEvent, DragEvent, PureComponent} from 'react';
import {connect} from 'react-redux';
import {push} from 'react-router-redux';
import {loadFile} from '../../lib/file-load/actions';
import {TraceFile} from '../../lib/file-load/file';
import {loadingFiles, validTraceFiles} from '../../lib/file-load/reducer';
import {AppState} from '../../lib/state/reducers';
import {Button, Glyphicon} from 'react-bootstrap';
import {Routes} from '../../lib/nav/routes';

interface StateProps
{
    files: TraceFile[];
    validTraceFiles: TraceFile[];
    loadingFiles: TraceFile[];
}

interface DispatchProps
{
    loadFile: (file: File) => {};
    goToNextPage: () => {};
}

class TraceLoaderComponent extends PureComponent<StateProps & DispatchProps>
{
    render()
    {
        return (
            <div>
                <input type='file' multiple={true} onChange={this.handleTraceChange} onDrop={this.handleTraceDrop} />
                <ul>
                    {this.props.files.map(this.renderFile)}
                </ul>
                <Button
                    disabled={!this.canGoToNextPage()}
                    onClick={this.props.goToNextPage}
                    bsStyle='primary'>
                    <Glyphicon glyph='flash' /> Load trace
                </Button>
            </div>
        );
    }

    renderFile = (file: TraceFile): JSX.Element =>
    {
        return (
            <li key={file.name}>
                <span>{file.name}, loading: {file.loading ? 'true' : 'false'}, error: {file.error}</span>
            </li>
        );
    }

    handleTraceChange = (event: ChangeEvent<HTMLInputElement>) =>
    {
        for (let i = 0; i < event.target.files.length; i++)
        {
            this.props.loadFile(event.target.files[i]);
        }
    }
    handleTraceDrop = (event: DragEvent<HTMLInputElement>) =>
    {
        const files = event.dataTransfer.files;
        for (let i = 0; i < files.length; i++)
        {
            this.props.loadFile(files[i]);
        }
    }

    canGoToNextPage = (): boolean =>
    {
        return this.props.validTraceFiles.length > 0 && this.props.loadingFiles.length < 1;
    }
}

export const TraceLoader = connect<StateProps, DispatchProps, {}>((state: AppState) => ({
    files: state.fileLoader.files,
    validTraceFiles: validTraceFiles(state),
    loadingFiles: loadingFiles(state)
}), ({
    loadFile: loadFile.started,
    goToNextPage: () => push(Routes.TraceVisualisation)
}))(TraceLoaderComponent);
