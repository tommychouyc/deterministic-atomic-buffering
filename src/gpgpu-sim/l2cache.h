// Copyright (c) 2009-2011, Tor M. Aamodt
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef MC_PARTITION_INCLUDED
#define MC_PARTITION_INCLUDED

#include "dram.h"
#include "../abstract_hardware_model.h"

#include <list>
#include <queue>

#define PUSH_PLACEHOLDER 0
#define PLACEHOLDER_SIZE 64

class mem_fetch;

class partition_mf_allocator : public mem_fetch_allocator {
public:
    partition_mf_allocator( const memory_config *config )
    {
        m_memory_config = config;
    }
    virtual mem_fetch * alloc(const class warp_inst_t &inst, const mem_access_t &access) const 
    {
        abort();
        return NULL;
    }
    virtual mem_fetch * alloc(new_addr_type addr, mem_access_type type, unsigned size, bool wr) const;
private:
    const memory_config *m_memory_config;
};

// Memory partition unit contains all the units assolcated with a single DRAM channel. 
// - It arbitrates the DRAM channel among multiple sub partitions.  
// - It does not connect directly with the interconnection network. 
class memory_partition_unit
{
public: 
   memory_partition_unit( unsigned partition_id, const struct memory_config *config, class memory_stats_t *stats );
   ~memory_partition_unit(); 

   bool busy() const;

   void cache_cycle( unsigned cycle );
   void dram_cycle();

   void set_done( mem_fetch *mf );

   void visualizer_print( gzFile visualizer_file ) const;
   void print_stat( FILE *fp ) { m_dram->print_stat(fp); }
   void visualize() const { m_dram->visualize(); }
   void print( FILE *fp ) const;
   void handle_memcpy_to_gpu( size_t dst_start_addr, unsigned subpart_id, mem_access_sector_mask_t mask );

   class memory_sub_partition * get_sub_partition(int sub_partition_id) 
   {
      return m_sub_partition[sub_partition_id]; 
   }

   // Power model
   void set_dram_power_stats(unsigned &n_cmd,
                             unsigned &n_activity,
                             unsigned &n_nop,
                             unsigned &n_act,
                             unsigned &n_pre,
                             unsigned &n_rd,
                             unsigned &n_wr,
                             unsigned &n_req) const;

   int global_sub_partition_id_to_local_id(int global_sub_partition_id) const; 

   unsigned get_mpid() const { return m_id; }

private: 

   unsigned m_id;
   const struct memory_config *m_config;
   class memory_stats_t *m_stats;
   class memory_sub_partition **m_sub_partition; 
   class dram_t *m_dram;

   class arbitration_metadata
   {
   public: 
      arbitration_metadata(const struct memory_config *config); 

      // check if a subpartition still has credit 
      bool has_credits(int inner_sub_partition_id) const; 
      // borrow a credit for a subpartition 
      void borrow_credit(int inner_sub_partition_id); 
      // return a credit from a subpartition 
      void return_credit(int inner_sub_partition_id); 

      // return the last subpartition that borrowed credit 
      int last_borrower() const { return m_last_borrower; } 

      void print( FILE *fp ) const; 
   private: 
      // id of the last subpartition that borrowed credit 
      int m_last_borrower; 

      int m_shared_credit_limit; 
      int m_private_credit_limit; 

      // credits borrowed by the subpartitions
      std::vector<int> m_private_credit; 
      int m_shared_credit; 
   }; 
   arbitration_metadata m_arbitration_metadata; 

   // determine wheither a given subpartition can issue to DRAM 
   bool can_issue_to_dram(int inner_sub_partition_id); 

   // model DRAM access scheduler latency (fixed latency between L2 and DRAM)
   struct dram_delay_t
   {
      unsigned long long ready_cycle;
      class mem_fetch* req;
   };
   std::list<dram_delay_t> m_dram_latency_queue;
};

class memory_sub_partition
{
public:
   memory_sub_partition( unsigned sub_partition_id, const struct memory_config *config, class memory_stats_t *stats );
   ~memory_sub_partition(); 

   unsigned get_id() const { return m_id; } 

   bool busy() const;

   void cache_cycle( unsigned cycle );

   bool full() const;
   bool full(unsigned size) const;
   void push( class mem_fetch* mf, unsigned long long clock_cycle );
   class mem_fetch* pop(); 
   class mem_fetch* top();
   void set_done( mem_fetch *mf );

   unsigned flushL2();
   unsigned invalidateL2();

   // interface to L2_dram_queue
   bool L2_dram_queue_empty() const; 
   class mem_fetch* L2_dram_queue_top() const; 
   void L2_dram_queue_pop(); 

   // interface to dram_L2_queue
   bool dram_L2_queue_full() const; 
   void dram_L2_queue_push( class mem_fetch* mf ); 

   void visualizer_print( gzFile visualizer_file );
   void print_cache_stat(unsigned &accesses, unsigned &misses) const;
   void print( FILE *fp ) const;

   void accumulate_L2cache_stats(class cache_stats &l2_stats) const;
   void get_L2cache_sub_stats(struct cache_sub_stats &css) const;

   // Support for getting per-window L2 stats for AerialVision
   void get_L2cache_sub_stats_pw(struct cache_sub_stats_pw &css) const;
   void clear_L2cache_stats_pw();

   void force_l2_tag_update(new_addr_type addr, unsigned time, mem_access_sector_mask_t mask)
   {
        m_L2cache->force_tag_access( addr, m_memcpy_cycle_offset + time, mask );
        m_memcpy_cycle_offset += 1;
   }

   std::bitset<40> cluster_inited;
   std::bitset<40> cluster_write_req;
   std::bitset<40> cluster_done;
   std::vector<int> remaining_addresses;
   std::vector<std::vector<mem_fetch*>> reorder_buffers;
   bool atomics;
   int cluster_serviced;
   int expected_flush_messages;

   std::vector<std::vector<unsigned>> remaining_addr_queue;

   new_addr_type queue_addr;

   unsigned long total_msgs;
   unsigned long msg_res_fails;
   unsigned long msg_misses;
   unsigned long msg_hits;
   unsigned long msg_hit_reserved;
   unsigned long long reordered_atomics;

    unsigned long long max_length_per_addr_queue[40];
    unsigned long long max_queue_length_total;

   unsigned long long max_length_per_buffer[40];
   unsigned long long max_length_total;

    unsigned long long atomic_total;
    unsigned long long atomic_hit;
    unsigned long long atomic_miss;
    unsigned long long atomic_pending_hit;
    unsigned long long atomic_res_fail;

    unsigned long long rest_total;
    unsigned long long rest_hit; 
    unsigned long long rest_miss;
    unsigned long long rest_pending_hit;
    unsigned long long rest_res_fail;

   //bool with_l2_write_queue;

   void print_reorder_stats()
   {
       printf("ID: %d\n", get_id());
       printf("Max Entries: %d\tNumber of re-orderings: %d\n",max_length_total, reordered_atomics);
       for (int i = 0; i < 40; i++)
       {
           printf("%d (%d)\t", i, max_length_per_buffer[i]);
       }
       printf("\n\ntotal_msgs=%llu\n", total_msgs);
       printf("msg_hits=%llu\n", msg_hits);
       printf("msg_misses=%llu\n",msg_misses);
       printf("msg_pending_hits=%llu\n", msg_hit_reserved);
       printf("msg_res_fails=%llu\n",msg_res_fails);
       
       printf("\n\ntotal_atomics=%llu\n", atomic_total);
       printf("atomic_hits=%llu\n", atomic_hit);
       printf("atomic_misses=%llu\n",atomic_miss);
       printf("atomic_pending_hits=%llu\n", atomic_pending_hit);
       printf("atomic_res_fails=%llu\n",atomic_res_fail);
       printf("atomic_miss_rate=%.2f\n", ((float) atomic_miss)/atomic_total);
       
       printf("\n\ntotal_rest=%llu\n", rest_total);
       printf("rest_hits=%llu\n", rest_hit);
       printf("rest_misses=%llu\n",rest_miss);
       printf("rest_pending_hits=%llu\n", rest_pending_hit);
       printf("rest_res_fails=%llu\n",rest_res_fail);
       printf("rest_miss_rate=%.2f\n", ((float) rest_miss)/rest_total);
       reordered_atomics = 0;
       printf("\n");
   }

    void log_reorder_stats()
    {
        int tot = 0;
        for (int i = 0; i < 40; i++)
        {
            tot += reorder_buffers[i].size();

            if (reorder_buffers[i].size() > max_length_per_buffer[i])
            {
                max_length_per_buffer[i] = reorder_buffers[i].size();
            }
        }

        if (tot > max_length_total)
        {
            max_length_total = tot;
        }
    }

    void print_addr_queue_stats()
   {
       printf("ID: %d\n", get_id());
       printf("Max Entries: %d\n",max_queue_length_total);
       for (int i = 0; i < 40; i++)
       {
           printf("%d (%d)\t", i, max_length_per_addr_queue[i]);
       }
       printf("\n");
   }

    void log_queue_stats()
    {
        int tot = 0;
        for (int i = 0; i < 40; i++)
        {
            tot += remaining_addr_queue[i].size();

            if (remaining_addr_queue[i].size() > max_length_per_addr_queue[i])
            {
                max_length_per_addr_queue[i] = remaining_addr_queue[i].size();
            }
        }
        if (tot > max_queue_length_total)
        {
            max_queue_length_total = tot;
        }
    }

   void set_next_cluster_serviced()
   {
       //printf("Partition %d ", get_id());
       for (int i = 0; i < 40; i++)
       {
           // look for next cluster to be serviced
           int tested_cluster = (cluster_serviced + i + 1)%40;
          // printf("%i (%d) ", tested_cluster, reorder_buffers[tested_cluster].size());
           if (remaining_addresses[tested_cluster] > 0)
           {
           // printf("\nCycle %d Partition %d moving %d to %d\n",0, get_id(), cluster_serviced, tested_cluster);
               cluster_serviced = tested_cluster;
               return;
           }
       }

       // all clusters done, check if there is already another set of flush messages queued up
        // check if all messages arrived for given flush
        queue_addr = 0xffffffff00000000;
        bool everything_arrived = true;
        for (int i = 0; i < 40; i++)
        {
            //if ((remaining_addr_queue[i].size() == 0 && !cluster_done[i]) || (remaining_addr_queue[i].front() > 0 && (remaining_addr_queue[i].front() > reorder_buffers[i].size())))
            if ((remaining_addr_queue[i].size() == 0 && !cluster_done[i]))
            {
                everything_arrived = false;
                break;
            }
        }

        // if so, start reordering atomics again
        if (everything_arrived)
        {
            // set remaining addr and first cluster
            bool expect_nothing = true;
            for (int i = 0; i < 40; i++)
            {
                assert(remaining_addr_queue[i].size() > 0 || cluster_done[i]);
                if (remaining_addr_queue[i].size() > 0)
                {
                    remaining_addresses[i] = remaining_addr_queue[i].front();
                    remaining_addr_queue[i].erase(remaining_addr_queue[i].begin());
                }
                else
                {
                    assert(cluster_done[i]);
                    remaining_addresses[i] = 0;
                }
            }
            for (int i = 0; i < 40; i++)
            {
                if (remaining_addresses[i] > 0)
                {
                    cluster_serviced = i;
                    expect_nothing = false;
                    break;
                }
            }
            atomics = !expect_nothing;
        }
        else
        {
            atomics = false;
        }

        /*
       atomics = 0;
       cluster_inited.reset();
       cluster_write_req.reset();
       expected_flush_messages = -1;*/
       //printf("\nCycle %d Partition %d Done\n",0, get_id());
   }

    bool push_atomic(unsigned long long cycle);

private:
// data
   unsigned m_id;  //< the global sub partition ID
   const struct memory_config *m_config;
   class l2_cache *m_L2cache;
   class L2interface *m_L2interface;
   partition_mf_allocator *m_mf_allocator;

   // model delay of ROP units with a fixed latency
   struct rop_delay_t
   {
    	unsigned long long ready_cycle;
    	class mem_fetch* req;
   };
   std::queue<rop_delay_t> m_rop;

   // these are various FIFOs between units within a memory partition
   fifo_pipeline<mem_fetch> *m_icnt_L2_queue;
   fifo_pipeline<mem_fetch> *m_L2_dram_queue;
   fifo_pipeline<mem_fetch> *m_dram_L2_queue;
   fifo_pipeline<mem_fetch> *m_L2_icnt_queue; // L2 cache hit response queue

   class mem_fetch *L2dramout; 
   unsigned long long int wb_addr;

   class memory_stats_t *m_stats;

   std::set<mem_fetch*> m_request_tracker;

   friend class L2interface;

   std::vector<mem_fetch*> breakdown_request_to_sector_requests(mem_fetch* mf);

   // This is a cycle offset that has to be applied to the l2 accesses to account for
   // the cudamemcpy read/writes. We want GPGPU-Sim to only count cycles for kernel execution
   // but we want cudamemcpy to go through the L2. Everytime an access is made from cudamemcpy
   // this counter is incremented, and when the l2 is accessed (in both cudamemcpyies and otherwise)
   // this value is added to the gpgpu-sim cycle counters.
   unsigned m_memcpy_cycle_offset;
};

class L2interface : public mem_fetch_interface {
public:
    L2interface( memory_sub_partition *unit ) { m_unit=unit; }
    virtual ~L2interface() {}
    virtual bool full( unsigned size, bool write) const 
    {
        // assume read and write packets all same size
        return m_unit->m_L2_dram_queue->full();
    }
    virtual void push(mem_fetch *mf) 
    {
        mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,0/*FIXME*/);
        m_unit->m_L2_dram_queue->push(mf);
    }
private:
    memory_sub_partition *m_unit;
};

#endif
