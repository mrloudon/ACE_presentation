import csv
import json
import numpy as np
import os
import pprint
from scipy import io
import time

# Marker channel threshold voltage
THRESHOLD = 2.5
# Maximum number of event codes
MAX_CODE = 255
# Tolerance (in samples) for an edge to be considered
# part of a particulr code
MAX_TOLERANCE = 10
# Max iterations of detectAllCodes (equals max num of emg events)
MAX_ITERATIONS = 10000


__all__ = ['load_config', 'load_data', 'analyse_data', 'analyse_event', 'analyse_events',
            'analyse_file', 'analyse_file_list', 'detect_all_codes', 'detect_all_edges',
            'determine_rms', 'fast_detect_edges', 'get_code_edges',
            'get_interval', 'load_config', 'load_data', 'process_rms_list', 'describe']

class EmgException(Exception): pass

def load_data(mat_file):
    """
    Loads an EMG data file exported by Biopak in MATLAB format.
    """
    def get_isi(isi, units):
        if units.lower() == 'ms':
            return isi * 1.0
        if units.lower() == 'us':
            return isi / 1000.0
        if units.lower() == 's':
            return isi * 1000.0
        raise EmgException('Unable to determine ISI.')
    
    raw = {}
    try:
        raw = io.loadmat(file_name=mat_file.__str__(), variable_names=['data', 'isi', 'isi_units', 'labels']) 
        isi = get_isi(raw['isi'][0][0], raw['isi_units'][0])
        raw['Isi'] = isi
    except (IOError, EmgException) as e:
        raise EmgException('Unable to load MATLAB file {0}\n{1}'.format(mat_file, e))
    return raw

def load_config(config_file):
    """
    Loads an EMG configuration file (expects JSON format).
    """
    try:
        json_data = open(config_file)
    except IOError as e:
        raise EmgException ('Unable to open {0}:\n\t{1}.'.format(config_file, e))

    try:
        j_data = json.load(json_data)
        emg_channel_set = j_data['EmgChannels']
        marker_channels = j_data['MarkerChannels']
        interval_set = j_data['IntervalSet']
        do_baseline = j_data['DoBaseline']
        marker_width = j_data['MarkerWidth']
        numeric_precision = j_data['NumericPrecision']
        event_codes = j_data['EventCodes']
        analysis_list = j_data['AnalysisList']
        if marker_width < 1:
            raise EmgException('Invalid marker width in configuration file: {0}.'.format(marker_width))
        if numeric_precision < 1 or numeric_precision > 20:
            raise EmgException('Invalid numeric precision in configuration file: {0}.'.format(numeric_precision))
        if analysis_list == []:
            raise EmgException('Empty analysis list in configuration file.') 
    except (ValueError, KeyError) as e:
        raise EmgException('Invalid values in configuration file: {0}\n\t{1}.'.format(a_file, e))
    finally:
        json_data.close()
  
    emg_channels = []
    emg_channel_labels = {}
    for channel_data in emg_channel_set:
        emg_channels.append(channel_data["Channel"])
        emg_channel_labels[channel_data["Channel"]] = channel_data["Label"]

    return   {
        'emgChannels': emg_channels,
        'emgChannelLabels': emg_channel_labels,
        'markerChannels': marker_channels, 
        'intervalSet': interval_set,
        'doBaseline': do_baseline,
        'markerWidth': marker_width,
        'numericPrecision': numeric_precision,
        'eventCodes': event_codes,
        'analysisList': analysis_list
    }
  
def describe(data):
    try:
        print("Header: ", data["__header__"].decode())
        print("Channels: ", data["data"].shape[1])
        print("Points per channel: ", data["data"].shape[0])
        print("Internal ISI: ", data['isi'][0][0],data['isi_units'][0])
        print("Interpreted as: ", data["Isi"]," ms (should be the same as Internal ISI above)")
        print("Biopak channel labels:")
        for chan, label in enumerate(data["labels"]):
            print("  ", chan, " ", label)
    except KeyError:
        print("Invalid key in MATLAB emg data")
    except TypeError:
        print("Invalid MATLAB emg data")

def fast_detect_edges(data = [], marker_width = 0):
    """
    High speed edge detector. Requires marker_width < marke pulse width to
    avoid missing edges. data[] is 1d list of volatges.
    """
    if data is None or len(data) == 0:
        raise EmgException('detectEdges() passed empty data array.')
    if marker_width < 1:
        raise EmgException('detectEdges() passed bad marker width.')
    n = 0
    off = True
    edges = []
    N = len(data)
    while n < N:
        if off and data[n] > THRESHOLD:
            off = False
            # found a high spot, now back back to find leading edge
            while n > 0:
                n -= 1
                if data[n] < THRESHOLD:
                    n += 1
                    edges.append(n)
                    break
        if not off and data[n] < THRESHOLD:
            off = True
        n += marker_width
    return edges

def detect_all_edges(data, marker_channels, marker_width, callback=None):
    """
    Returns an edge set (list of lists of edges) given the raw data and a list of marker channels.
    The optional callback function returns progress information as this can be a slow process.
    """
    if data is None or len(data) == 0:
        raise EmgException('detect_all_edges() passed empty data array.')
    if marker_channels == None or len(marker_channels) == 0:
        raise EmgException('detect_all_edges() passed empty marker channel array.')
    edge_set = []
    for marker in marker_channels:
        try:
            if callback:
                callback('Edge detecting marker channel: {0}.'.format(marker))
            edge_set.append(fast_detect_edges(data = data[:, marker-1], marker_width = marker_width))
        except IndexError:
            raise EmgException('Index error in detectAllEdges(). The marker channels are inconsistent with the data array.')
    return edge_set

def detect_all_codes(edge_set):
    """
    Returns a dictionary consisting of a list of all the event codes from an edge set (a list of lists of edges)
    and a set of unique codes.
    Note: trashes edge_set!!
    """
    # Return False if there are any non-empty edge lists in edge set.
    def exhausted():
        for edge_list in edge_set:
            if len(edge_list) > 0:
                return False
        return True
    
    # Return the index of the edge list containing the lowest (first) edge.
    # The lists are ordered so we can assume that the lowest value in the 
    # list is also the first value in the list.
    def get_first():
        current_index = 0
        result = None
        lowest = None
        for edge_list in edge_set:
            # Skip empty edge lists
            if len(edge_list) > 0:
                if lowest is None:
                    # initial lowest is the first
                    lowest = edge_list[0]
                    result = current_index
                else: 
                    if edge_list[0] < lowest:
                        lowest = edge_list[0]
                        result = current_index
            current_index += 1
        if result is None:
            raise EmgException('get_first() failed. Unable to identify first edge of edge_set.')
        return result
    
    # Return the current binary code value and its edge in a tupple code, edge
    def get_current_code_and_edge():
        edge = edge_set[get_first()][0]
        max_tol = edge + MAX_TOLERANCE
        mult = 1
        code = 0
        for edge_list in edge_set:
            if len(edge_list) > 0:
                if edge_list[0] >= edge and edge_list[0] < max_tol:
                    code += mult
                    edge_list.pop(0)
            mult *= 2
        return (code, edge)

    code_and_edges = []
    code_set = set()
    n = 0
    while not exhausted():
        a_code_and_edge  = get_current_code_and_edge()
        code_and_edges.append(a_code_and_edge)
        code_set.add(a_code_and_edge[0])
        n += 1
        if n > MAX_ITERATIONS:
             raise EmgException('Caught in infinite loop in detect_all_codes()')
    return {'codeAndEdgeList': code_and_edges, 'codeSet': code_set}

def get_code_edges(code_and_edge_list, code):
    """
    Returns a list of edges corresponding to a given code from the master list of all codes and edges
    """
    if code_and_edge_list is None or len(code_and_edge_list) == 0:
        raise EmgException('Empty code and edge list passed to get_code_edges()')
    edges = []
    for ce in code_and_edge_list:
        if ce[0] == code:
            edges.append(ce[1])
    return edges

def get_interval(data, emg_channel, edge, interval):
    """
    Returns the data found in a given channel over a given interval positioned by an edge.
    """
    if len(interval) < 2:
        raise EmgException('Invalid interval passed to getInterval(): Start and stop not found.')
    start = edge + interval[0]
    stop = edge + interval[1]
    interval_data = data[start:stop, emg_channel-1]
    if interval_data == []:
        raise EmgException('Error from get_interval(). No data in interval: start = {0}, stop = {1}'.format(start, stop))
    return interval_data   

def determine_rms(data, emg_channel, edge, interval_list, do_baseline, trial_num, channel_labels):
    """
    Returns a list of RMS values for the given EMG channel and edge over the list of intervals.
    """
    rms = []
    labels = []
    n = 1
    for interval in interval_list:
        dat = get_interval(data = data, emg_channel = emg_channel, edge = edge, interval = interval)
        rms.append(np.sqrt(np.mean(dat**2)))
        if do_baseline:
            if n == 1: # Baseline
                labels.append('{0}_b_{1}_'.format(trial_num, 
                                                 channel_labels[emg_channel]))
            else: # Not baseline but a t interval one less than n
                labels.append('{0}_t{1}_{2}_'.format(trial_num, n - 1, 
                                                 channel_labels[emg_channel]))
        else: # No baseline, only t values
            labels.append('{0}_t{1}_{2}_'.format(trial_num, n, 
                                                 channel_labels[emg_channel]))
        n += 1
    return rms, labels

def process_rms_list(rms_list = [], rms_labels = [], emg_channel = -1, do_baseline = True, trial_num = 0, channel_labels = []):
    """
    Performs the normalisation.
    If do_baseline is True, return a dictionary containing the rms_list,
    normalised value list, and a csv representation of the data.
    Else return the rms_list and csv representation only with an
    empty normalised list.
    """
    if rms_list == [] or rms_labels == []:
        raise EmgException('No RMS values/labels to analyse: analyse_rms_list()')
    if emg_channel < 1:
        raise EmgException('Invalid EMG channel: analyse_rms_list()')
    if channel_labels == []:
        raise EmgException('Invalid channel labels: analyse_rms_list()')
    # no baseline so just return the rms values
    if not do_baseline:
        return {
                    'rms': rms_list, 'rmsLabels': rms_labels, 'normalised': [], 
                    'normalisedLabels': []
               }
    else:
        base = rms_list[0]
        normalised_list = []
        normalised_labels = []
        if base == 0.0:
            raise EmgException('Baseline is zero: analyseRmsList()')
        if len(rms_list) < 2:
            raise EmgException('Not enough RMS values to analyse baseline normalisation: analyseRmsList()')
        for n in range(len(rms_list)-1):
            norm = rms_list[n+1] / base
            normalised_list.append(norm)
            normalised_labels.append('{0}_n{1}_{2}_'.format(trial_num, n+1, channel_labels[emg_channel]))
        return {
                    'rms': rms_list, 'rmsLabels': rms_labels, 'normalised': normalised_list, 
                    'normalisedLabels': normalised_labels
                } 


def analyse_event(data, emg_channel_list, edge, interval_list, do_baseline, trial_num, emg_channel_labels):
    """
    Analyse a particular event given by edge over an interval list and emg channel list.
    Return a dictionary of the analysed data. Event defines the pivot of the interval list.
    """
    analysed_data = []
    all_labels = []
    for emg_channel in emg_channel_list:
        rms_data, labels = determine_rms(data = data, emg_channel = emg_channel, 
            edge = edge, interval_list = interval_list, do_baseline = do_baseline,
            trial_num = trial_num, channel_labels = emg_channel_labels)
        result = process_rms_list(rms_list = rms_data, rms_labels = labels, 
            emg_channel = emg_channel, do_baseline = do_baseline, 
            trial_num = trial_num, channel_labels = emg_channel_labels)
        analysed_data = analysed_data + result['rms'] + result['normalised']
        all_labels = all_labels + result['rmsLabels'] + result['normalisedLabels']
    return {'analysedData': analysed_data, 'allLabels': all_labels}

def analyse_events(data, emg_channel_list, edge_list, interval_list, do_baseline, event_type_label, emg_channel_labels):
    """
    Analyse a list of events given by an edge list over and interval list and emg channel list.
    Return a dictionary of the analysed data.
    """
    analysed_data = []
    all_labels = []
    n = 1
    for edge in edge_list:
        result = analyse_event(data, emg_channel_list, edge, interval_list, do_baseline, n, emg_channel_labels )
        analysed_data = analysed_data + result['analysedData']
        all_labels = all_labels + result['allLabels']
        n += 1
    for n in range(len(all_labels)):
        all_labels[n] = event_type_label + all_labels[n]

    return {'analysedData': analysed_data, 'allLabels': all_labels}

def analyse_data(raw_data, config, callback = None):
    """
    The high level analysis call. Analyses a raw matlab file (with the additional Isi key appended) against
    a global conifiguration dictionary. Assumes configuration intervals are in ms (not samples).
    The function callback() will be called regularly with a text argument to provide feedback on progress
    and any errors.
    """
    # Calls callback if not None
    def log(msg):
        if callback:
            callback(msg)

    # Convert an interval list (i.e. a list of lists) expressed in milliseconds to an interval list 
    # expressed in terms of  the number of (integer) samples.
    def convert_ms_intervals_to_sample_intervals(ms_intervals):
        isi = raw_data['Isi']
        sample_intervals = []
        for ms_interval in ms_intervals:
            sample_interval = []
            for n_ms in ms_interval:
                sample_interval.append(int((1.0 * n_ms) / isi))
            sample_intervals.append(sample_interval)
        return sample_intervals

    # Converts sample list (single list of sample values) to a list of second values
    def convert_sample_list_to_sec_list(sampleList):
        isi = raw_data['Isi']
        sec_edges = []
        for sample in sampleList:
            sec_edges.append(sample * 0.001 * isi)
        return sec_edges

    # Returns a sample interval set converted from the configuration's millisecond interval set.
    def get_sample_interval_set():
        interval_set = config['intervalSet']
        for interval_dict in interval_set:
            sample_interval_list = convert_ms_intervals_to_sample_intervals(interval_dict['IntervalList'])
            interval_dict['IntervalList'] = sample_interval_list
        return interval_set

    log('Isi: {0} ms\n'.format(raw_data['Isi']))
    log('Interval set expressed in ms: \n{0}\n'.format(pprint.pformat(config['intervalSet'])))
    interval_set = get_sample_interval_set()
    log('Interval set expressed in samples: \n{0}\n'.format(pprint.pformat(interval_set)))
    
    # First detect all the edges
    edge_set = detect_all_edges(raw_data['data'], config['markerChannels'], config['markerWidth'], log)
    # Then reduce the detected edges to codes and corresponding edges where the codes apply
    log('\nDetermining event codes from edge set.')
    extracted_codes_and_edges = detect_all_codes(edge_set)
    log('\nUnique codes determined from edge set:\n{0}\n'.format(pprint.pformat(extracted_codes_and_edges['codeSet'])))
    log('\nCodes to be analysed:\n{0}\n'.format(pprint.pformat(config['analysisList'])))
    for code in extracted_codes_and_edges['codeSet']:    
        # codeEdges are in samples not ms/seconds
        code_edges = get_code_edges(extracted_codes_and_edges['codeAndEdgeList'], code)
        # convert samples number to seconds
        code_edges_sec = convert_sample_list_to_sec_list(code_edges)
        # And format the floats down to one dp for printing
        pretty = []
        for sec in code_edges_sec:
            pretty.append('{0:.1f}'.format(sec))
        try:
            log('Code: {0}, Label: {1}, N: {2}\n{3}'.format(code, config['eventCodes'][str(code)],len(pretty),pprint.pformat(pretty, width=80)))
        except KeyError:
            log('\nKey error: WARNING an unexpected code was encountered: {0}\n'.format(str(code)))
            config['eventCodes'][str(code)] = 'unknown'
            log('Code: {0}, Label: {1}\n{2}'.format(code, config['eventCodes'][str(code)] ,pprint.pformat(pretty, width=80)))

    emg_channels = config['emgChannels']
    #log('Emg channels: {0}\n'.format(pprint.pformat(emgChannels)))
    log('Emg channels: {0}\n'.format(pprint.pformat(config['emgChannelLabels'])))
    all_data = []
    all_labels = []

    for n in range(len(interval_set)):
        intervals = interval_set[n]['IntervalList']
        log('\nDoing interval list: {0}'.format(intervals))
        for event_code in config['analysisList']:
            if not event_code in extracted_codes_and_edges['codeSet']:
                log('\nWARNING: Skipping an unexpected code that was encountered in the AnalysisList: {0}\n'.format(str(event_code)))
                continue
            label = config['eventCodes'][str(event_code)]
            edges = get_code_edges(extracted_codes_and_edges['codeAndEdgeList'], event_code)
            analysis_result = analyse_events(raw_data['data'], emg_channels, edges, intervals, config['doBaseline'], label, config['emgChannelLabels'])
            for n_label in range(len(analysis_result['allLabels'])):
                analysis_result['allLabels'][n_label] += interval_set[n]['Label']
            all_data.extend(analysis_result['analysedData'])
            all_labels.extend(analysis_result['allLabels'])

    return { 'analysedData': all_data, 'dataLabels': all_labels }

def analyse_file(mat_file, config, callback = None):
    all_data = []
    all_labels = []
    try:
        # Pre-append the matlab file name to the data list
        m_file_name = os.path.split(mat_file)[1]
        all_data.append(m_file_name)
        # and the matlab file's time date stamp
        td_stamp = time.ctime(os.path.getmtime(mat_file))
        all_data.append(td_stamp)
        # and append appropriate variable labels
        all_labels.append('file_name')
        all_labels.append('time_date')
    except IOError as e:
        log(e)
        raise EmgException('Unable to access: {0}'.format(mat_file))

    raw = load_data(mat_file)
    result = analyse_data(raw, config, callback)
    all_data.extend(result['analysedData'])
    all_labels.extend(result['dataLabels'])

    return { 'allData': all_data, 'allLabels': all_labels, 'matFileName': m_file_name, 'timeDateStamp': td_stamp }

def analyse_file_list(file_list, config_file, output_file, callback):
    """
    Iterates over matlab fileList using config_file.
    Analysis results accumulate in output_file.
    """

    config = load_config(config_file)

    first_file = True
    
    with open(output_file, 'w', newline = '') as csv_file:
        master_results = { 'variables': None, 'data': []}
        w = csv.writer(csv_file, delimiter = ',')
        for matlab_file in file_list:
            # The actual data
            formatted_data = []
            # The variable names
            #formatted_variables = []
            callback('\n--------------------------------------------------------------------------------')
            callback('\nAnalysing file: \n{0}\n'.format(matlab_file));
            result = analyse_file(matlab_file, config, callback)
            
            for datum in result['allData']:
                # datum could be the filename, time stamp, rms voltage or normalised to baseline voltage
                if type(datum) is str:
                    # If its a string (file name, time date stamp, etc) leave it alone
                    formatted_data.append(datum)
                else:
                    # Then it should be a numpy float64, so format it so SPSS doesn't break
                    # Construct the format string for numeric data
                    p_str = '{0:.' + str(config['numericPrecision']) + 'f}'
                    formatted_data.append(p_str.format(datum))
            if first_file:
                first_file = False
                w.writerow(result['allLabels'])
                master_results["variables"] = result['allLabels']
            # Include the actual mat file name and TD stamp in the variable data
            #formatted_variables.append(result['matFileName'])
            #formatted_variables.append(result['timeDateStamp'])
            # Miss out the variable names for the mat file name and TD stamp
            #formatted_variables.extend(result['allLabels'][2:])
            w.writerow(formatted_data)
            master_results["data"].append(formatted_data)
            callback('\nN data items: {0}\n'.format(len(result['allData']) - 2))
        return master_results
    

##################################
# Module test code
##################################

def progress_callback(txt):
    print(txt)

if __name__ == "__main__":
    print("\nEMG utilities test\n\n")
    print("Loading EMG test file...\n")
    emg_data = load_data(mat_file = ".\\data\\P7.mat")
    pprint.pprint(emg_data)
    print("\nLoading configuration test file...\n")
    config = load_config(config_file = ".\\config\\malc.json")
    pprint.pprint(config)
    print("\nEdge detecting...\n")
    edge_set = detect_all_edges(emg_data["data"], config["markerChannels"], config["markerWidth"], progress_callback)
    pprint.pprint(edge_set)
    print("\nDetecting codes...\n")
    all_codes = detect_all_codes(edge_set)
    pprint.pprint(all_codes)
    code = 32
    print("\nDetecting code ", code)
    code_edges = get_code_edges(all_codes["codeAndEdgeList"], code)
    print(code_edges)
    print("\nGetting a 40ms interval...")
    interval = get_interval(emg_data["data"], 1, code_edges[5], (-20, 20))
    print (interval)
    print ("\nGetting RMS..\n")
    rms = determine_rms(emg_data["data"], 1, code_edges[5], [(-20,20), (0, 20)], True, 1, config["emgChannelLabels"])
    pprint.pprint(rms)
    print ("\nGetting normalized...\n")
    norm = process_rms_list(rms_list = rms[0], rms_labels = rms[1], emg_channel = 1, 
        do_baseline = True, trial_num = 1, channel_labels = config["emgChannelLabels"])
    pprint.pprint(norm)
    print("\nAnalyse event {0}...\n".format(code))
    event_analysis = analyse_event(data = emg_data["data"], emg_channel_list = config["emgChannels"], 
        edge = code_edges[5], interval_list = [(-20, 20), (0, 20)], do_baseline = True, 
        trial_num = 1, emg_channel_labels = config["emgChannelLabels"])
    pprint.pprint(event_analysis)
    
    print ("\nPerforming file analysis...")
    results = analyse_file_list(file_list = [".\\data\\P7.mat",".\\data\\P3.mat"], config_file = ".\\config\\malc.json", output_file = "test.csv", callback = progress_callback)
    print ("\nVariables:")
    #pprint.pprint(results["analysedData"])
    pprint.pprint(results['variables'][0:10])
    print ("\nData:")
    pprint.pprint(results['data'][0][0:10])
    pprint.pprint(results['data'][1][0:10])
    print("\nDone.")