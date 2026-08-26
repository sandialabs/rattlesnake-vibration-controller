# Load in data

filters = {
    # 'environment_queue':['Sine put','Sine got','Sine Command Queue'],
    # 'environment':['Sine'],
    # 'spectral':['Spectral Processing'],
    # 'siggen':['Signal Generation'],
    # 'collector':['Data Collector'],
    # 'analysis':['Data Analysis'],
    # 'acquisition':['Acquisition'],
    # 'output':['Output'],
    # 'streaming':['Streaming','STREAMING'],
    # 'sysid_streaming':['INITIALIZE_STREAMING','FINALIZE_STREAMING'],
    # 'ui':['UI '],
    # 'all':None
    # 'output_rms':['Arming','Output Writing Data to Hardware RMS']
    'acquisition_output':['Acquisition','Output']
           }

filter_operation = 'or'

lines_to_read = None

for name,log_filters in filters.items():
    print(name)
    with open('../Rattlesnake.log','r') as f:
        if lines_to_read is None:
            lines = f.readlines()
        else:
            lines = [f.readline() for i in range(lines_to_read)]
        
    dates = set([])
    for line in lines:
        dates.add(line[:26])
    dates = sorted(list(dates))

    with open('../debug_data/Rattlesnake_Ordered_{:}.log'.format(name),'w') as f:
        for date in dates:
            for line in lines:
                if filter_operation == 'and':
                    selection = (log_filters is None or all([log_filter in line for log_filter in log_filters]))
                elif filter_operation == 'or':
                    selection = (log_filters is None or any([log_filter in line for log_filter in log_filters]))
                if line[:26] == date and selection:
                    f.write(line.replace('////','\n'))