"""
Helper script to train trajectory models from data collected with the gazebo simulator (using get_data.py script)
"""

import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser(description="Train trajectory linear models from gazebo simulator data.")
parser.add_argument("--data_file", type=str, required=True, help="Path to the data.json file")

def preprocess_data(data):
    def normalize_angle(angle_in_radians):
        remainder = angle_in_radians % (2 * np.pi)
        if remainder > np.pi:
            remainder -= 2 * np.pi
        return remainder
    
    d = []
    for t, d_ in enumerate(data):
        
        # precompute some variables
        vels_agent = np.array(d_['vel_agent'])
        vels_land  = np.array(d_['vel_landmark'])
        vel_agent = np.linalg.norm(vels_agent, axis=1)
        vel_landmark = np.linalg.norm(vels_land, axis=1)
        angle_agent = np.arctan2(vels_agent[:, 1], vels_agent[:, 0])
        angle_land = np.arctan2(vels_land[:, 1], vels_land[:, 0])
        
        for i in range(len(d_['pos_agent'])):
            # agent data
            d.append({
                'step':i+1,
                'traj_id':t,
                'type':'agent',
                'dt':d_['dt'],
                'prop':d_['prop_agent'][i],
                'rudder':d_['rud_agent'][i],
                'depth':d_['pos_agent'][i][-1],
                'pos_x':d_['pos_agent'][i][0],
                'pos_y':d_['pos_agent'][i][1],
                'vel_x':d_['vel_agent'][i][0],
                'vel_y':d_['vel_agent'][i][1],
                'vel': vel_agent[i],
                'vel_x_start':np.nan if i==0  else d_['vel_agent'][i-1][0],
                'vel_y_start':np.nan if i==0  else d_['vel_agent'][i-1][1],
                'dir_start': np.nan if i==0 else angle_agent[i-1],
                'dir_end': angle_agent[i],
                'dir_diff': np.nan if i==0 else normalize_angle(angle_agent[i]-angle_agent[i-1]),
            })
            # landmark data
            d.append({
                'step':i+1,
                'traj_id':t,
                'type':'landmark',
                'dt':d_['dt'],
                'prop':d_['prop_land'][i],
                'rudder':d_['rud_land'][i],
                'depth':d_['pos_landmark'][i][-1],
                'pos_x':d_['pos_landmark'][i][0],
                'pos_y':d_['pos_landmark'][i][1],
                'vel_x':d_['vel_landmark'][i][0],
                'vel_y':d_['vel_landmark'][i][1],
                'vel': vel_landmark[i],
                'vel_x_start':np.nan if i==0 else d_['vel_landmark'][i-1][0],
                'vel_y_start':np.nan if i==0 else d_['vel_landmark'][i-1][1],
                'dir_start':np.nan if i==0 else angle_land[i-1],
                'dir_end': angle_land[i],
                'dir_diff': np.nan if i==0 else normalize_angle(angle_land[i]-angle_land[i-1]),
            })
            
    df = pd.DataFrame(d)
    return df

def linear_mae_and_plot(X, y, x_label, y_label, title):

    # mae
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f"{title} MAE:", np.mean(-scores))
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    print(f"{title} R2:", r2)

    # plot
    plot = False
    if plot:
        plt.clf()
        plt.plot(X,y, label='data')
        plt.plot(X,model.predict(X), label='pred')
        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.annotate(f"R2: {r2:.2f}\nMae: {np.mean(-scores):.2f}", xy=(0.7, 0.9), xycoords='axes fraction', fontsize=12)
        plt.savefig(f'{title}_model.png')
        plt.show()

    return model.coef_.item(), model.intercept_


def main():
 
    args = parser.parse_args()
    data_file_path = args.data_file
    with open(data_file_path, 'r') as f:
        data = json.load(f) 

    df = preprocess_data(data)

    # check the linear relationship between propulsor and velocity and save the model
    df_ = df.sort_values('prop')
    X = df_[['prop']].values
    y = df_['vel']
    coef, interc = linear_mae_and_plot(X, y, 'propulsor', 'linear_velocity', 'prop_vel')
    models = {'vel':{'coeff':coef,'intercept':interc}}
    
    
    # check the linear relationship between rudder and change of direction
    # here we group by dt, linear velocity and type (agent/landmark) because
    # the dyanmics are very different for each combination of those
    # save a different model for each combination of dt and velocity
    df_ = df[(df['dir_start'].notnull())&(df['type']=='agent')].copy() # consider only agents
    models['angle'] = {f'dt_{dt}':{} for dt in df_['dt'].unique()}
    for (dt, prop, type_), g in df_.groupby(['dt', 'prop','type']):
        g = g.sort_values('rudder')
        X = g[['rudder']].values
        y = g['dir_diff'].values
        right_idx = int(len(y)*0.9)
        left_idx = int(len(y)*0.1)
        y[right_idx:] = np.where(y[right_idx:]>0, y[right_idx:]-2*np.pi, y[right_idx:])
        y[:left_idx] = np.where(y[:left_idx]<0, y[:left_idx]+2*np.pi, y[:left_idx])
        title = f'rudder_angle_(prop_vel_{prop}_dt_{dt}_type_{type_})'
        coef, interc = linear_mae_and_plot(X, y, 'rudder', 'angle_change', title)
        models['angle'][f'dt_{dt}'][f'prop_{prop}'] = {'coeff':coef,'intercept':interc}

    with open('traj_linear_models_many.json', 'w') as f:
        json.dump(models, f)
        

if __name__=='__main__':
    main()
    
