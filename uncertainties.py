import matplotlib.pyplot as plt
import boosted_fits as bsvj

scripter = bsvj.Scripter()
@scripter
def systematics():
    """
    Fitting the systematics 
    """
    rootfile = bsvj.pull_arg('-i','--rootfile', type=str, help='input datacard rootfile').rootfile
    with bsvj.open_root(rootfile) as f:
      ws = bsvj.get_ws(f)
    systematics_list = ['isr','fsr','pu','pdf','jer','jec','jes_both','scale']
    syst={}
    syst_data={}
    sig = ws.data('sig')
    sig_data = bsvj.roodataset_values(sig)
    for i in systematics_list:
      syst[i]={}
      syst_data[i]={}
      print(i)
      for j in ['Up','Down']:
        syst[i][j] = ws.data('sig_'+str(i)+str(j))
        syst_data[i][j] = bsvj.roodataset_values(syst[i][j])
        plt.hist(syst_data[i][j][0],bins=52,weights=syst_data[i][j][1], histtype='step',linewidth=2,label=str(i)+str(j))
      plt.hist(sig_data[0],bins=52,weights=sig_data[1], histtype='step',linewidth=2,label='sig')
      plt.legend()
      plt.ylabel('A.U.')
      #plt.yscale('log')
      plt.xlabel('mT (GeV)')
      plt.savefig('systematics/'+str(i)+'.png')
      plt.close()
      
if __name__ == '__main__':
    scripter.run()
